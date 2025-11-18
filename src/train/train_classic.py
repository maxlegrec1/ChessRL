#!/usr/bin/env python3
"""
Supervised training script for the small BT4 model using classic policy/value data.

Loads (FEN, move, value distribution) batches from a parquet file and optimizes:
- Cross-entropy loss for the policy logits
- Mean-squared-error loss between the value_q head probabilities and target values

Key defaults (overridable via CLI):
- Data source: adversarial_moves.parquet
- Pretrained weights: src/models/small/model.pt
- Optimizer: Adam, lr=1e-5
- Batch size: 1024
- Steps: 1000
- Mixed precision (AMP) on CUDA
- Metrics logged to Weights & Biases
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.models.small.fen_encoder import batch_fen_to_tensor
from src.models.small.model import BT4
from src.models.small.vocab import policy_to_idx


@dataclass(frozen=True)
class TrainingSample:
    """Lightweight container for a single supervised sample."""

    fen: str
    policy_idx: int
    value_target: torch.Tensor  # Shape: [3] (black_win, draw, white_win)


def load_training_samples(parquet_path: Path) -> List[TrainingSample]:
    """
    Load supervised samples from the parquet file and filter out invalid entries.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(parquet_path, columns=["fen", "move", "value_json"])
    fens = table.column("fen").to_pylist()
    moves = table.column("move").to_pylist()
    values = table.column("value_json").to_pylist()

    samples: List[TrainingSample] = []
    dropped = 0

    for fen, move, value_json in zip(fens, moves, values):
        if not fen or not move:
            dropped += 1
            continue

        policy_idx = policy_to_idx.get(move)
        if policy_idx is None:
            dropped += 1
            continue

        if value_json is None:
            dropped += 1
            continue

        if isinstance(value_json, str):
            value_list = json.loads(value_json)
        else:
            value_list = list(value_json)

        if not isinstance(value_list, (list, tuple)) or len(value_list) != 3:
            dropped += 1
            continue

        value_tensor = torch.tensor(value_list, dtype=torch.float32)
        fen_parts = fen.split()
        if len(fen_parts) > 1 and fen_parts[1] == "w":
            value_tensor = value_tensor[[2, 1, 0]]
        total = float(value_tensor.sum().item())
        if total <= 0:
            dropped += 1
            continue
        value_tensor = value_tensor / total

        samples.append(TrainingSample(fen=fen, policy_idx=policy_idx, value_target=value_tensor))

    if not samples:
        raise RuntimeError(f"No valid samples were loaded from {parquet_path}")

    random.shuffle(samples)
    print(f"Loaded {len(samples)} samples from {parquet_path} (dropped {dropped} invalid rows)")
    return samples


def sample_batch(samples: Sequence[TrainingSample], batch_size: int) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Sample a batch (with replacement) of FENs, policy indices, and value targets."""
    idxs = np.random.randint(0, len(samples), size=batch_size)
    selected = [samples[i] for i in idxs]
    fens = [s.fen for s in selected]
    policy_indices = torch.tensor([s.policy_idx for s in selected], dtype=torch.long)
    value_targets = torch.stack([s.value_target for s in selected])
    return fens, policy_indices, value_targets


def forward_pass(model: BT4, board_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Execute the manual forward pass used during training to obtain policy logits
    and value_q logits.
    """
    bsz, seq_len, _, _, emb = board_batch.size()
    x = board_batch.view(bsz * seq_len, 64, emb)
    x = model.linear1(x)
    x = torch.nn.GELU()(x)
    x = model.layernorm1(x)
    x = model.ma_gating(x)

    pos_enc = model.positional(x)
    for layer in model.layers:
        x = layer(x, pos_enc)

    value_h_q = model.value_head_q(x).view(bsz, seq_len, 3)

    policy_tokens = model.policy_tokens_lin(x)
    policy_tokens = torch.nn.GELU()(policy_tokens)
    policy_tokens = policy_tokens + pos_enc
    queries = model.queries_pol(policy_tokens)
    keys = model.keys_pol(policy_tokens)
    attn_logits = torch.matmul(queries, keys.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(model.d_model)
    attn_logits = attn_logits.view(bsz, seq_len, 64 * 64)
    policy_logits = model.policy_head(attn_logits)
    return policy_logits, value_h_q


def parse_args() -> argparse.Namespace:
    default_parquet = REPO_ROOT / "adversarial_moves.parquet"
    default_weights = REPO_ROOT / "model_adversarial_last.pt"

    parser = argparse.ArgumentParser(description="Supervised training for the small BT4 model.")
    parser.add_argument("--parquet-file", type=Path, default=default_parquet, help="Path to training parquet file.")
    parser.add_argument("--model-weights", type=Path, default=default_weights, help="Path to pretrained model weights.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--steps", type=int, default=52, help="Number of optimizer steps to run.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--wandb-project", type=str, default="ChessRL", help="Weights & Biases project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity.")
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional list of W&B tags.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=REPO_ROOT / "model_adversarial_last_2.pt",
        help="Path to save the final checkpoint.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure deterministic-ish behavior
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    mixed_precision = device.type == "cuda"

    samples = load_training_samples(args.parquet_file)
    if len(samples) < args.batch_size:
        print(
            f"Warning: dataset has {len(samples)} samples which is smaller than the batch size "
            f"{args.batch_size}. Sampling will use replacement."
        )

    model = BT4().to(device)
    state_dict = torch.load(args.model_weights, map_location=device,weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=mixed_precision)

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            tags=args.wandb_tags,
            config={
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "parquet_file": str(args.parquet_file),
                "model_weights": str(args.model_weights),
                "device": args.device,
                "optimizer": "Adam",
                "mixed_precision": mixed_precision,
            },
        )

    for step in range(1, args.steps + 1):
        fens, policy_targets, value_targets = sample_batch(samples, args.batch_size)

        boards = batch_fen_to_tensor(fens, device=device.type, dtype=torch.float32).to(device)
        boards = boards.unsqueeze(1)  # (B, 1, 8, 8, 19)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True, device_type="cuda"):
            policy_logits, value_logits = forward_pass(model, boards)
            policy_logits = policy_logits[:, 0, :]
            value_logits = value_logits[:, 0, :]

            value_probs = torch.softmax(value_logits, dim=-1)
            policy_loss = F.cross_entropy(policy_logits, policy_targets)
            value_loss = F.mse_loss(value_probs, value_targets)
            total_loss = policy_loss + value_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            policy_pred = torch.argmax(policy_logits, dim=-1)
            policy_accuracy = (policy_pred == policy_targets).float().mean()

            value_pred = torch.argmax(value_probs, dim=-1)
            value_target_idx = torch.argmax(value_targets, dim=-1)
            value_accuracy = (value_pred == value_target_idx).float().mean()

            log_probs = F.log_softmax(policy_logits, dim=-1)
            probs = log_probs.exp()
            policy_entropy = (-(probs * log_probs).sum(dim=-1)).mean()

        metrics = {
            "loss/total": total_loss.detach().item(),
            "loss/policy": policy_loss.detach().item(),
            "loss/value": value_loss.detach().item(),
            "metrics/policy_accuracy": policy_accuracy.item(),
            "metrics/value_accuracy": value_accuracy.item(),
            "metrics/policy_entropy": policy_entropy.item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "global_step": step,
        }

        if wandb_run is not None:
            wandb.log(metrics, step=step)

        if step % 10 == 0 or step == 1 or step == args.steps:
            print(
                f"[Step {step:04d}/{args.steps}] "
                f"loss={metrics['loss/total']:.4f} "
                f"policy_acc={metrics['metrics/policy_accuracy']:.3f} "
                f"value_acc={metrics['metrics/value_accuracy']:.3f} "
                f"entropy={metrics['metrics/policy_entropy']:.3f}"
            )

    checkpoint_path: Path = args.checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": args.steps,
            "config": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

