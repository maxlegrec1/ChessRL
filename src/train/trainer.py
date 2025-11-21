from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from rich.console import Console
from torch import amp, optim

import wandb

from src.models.small.model import BT4

from .config import TrainingConfig
from .dataset import NULL_POLICY_INDEX, SelfPlayBatch, SelfPlayDataset


AMP_DTYPES: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.console = Console()
        self.device = torch.device(config.device)
        self.device_type = config.device
        self.dataset = SelfPlayDataset(
            config.data.dataset_path,
            reload_seconds=config.data.reload_seconds,
        )

        self.model = BT4().to(self.device)
        self.model.train()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        self.scaler = amp.GradScaler(device = self.device)

        self.amp_dtype = AMP_DTYPES.get(config.amp_dtype.lower(), torch.float16)
        self.checkpoint_dir = config.checkpoint.dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.samples_processed = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self._current_stockfish_elo = 1400

        self._wandb_run = self._init_wandb()
        if config.resume_path:
            self._load_checkpoint(config.resume_path)

    # ------------------------------------------------------------------ #
    def _init_wandb(self):
        if not self.config.wandb.enabled:
            return None
        run = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.wandb.run_name or self.config.run_name,
            config=self.config.to_dict(),
        )
        return run

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.console.log(
            f"Starting training on {self.device} with dataset size {self.dataset.size}"
        )
        try:
            while True:
                self.dataset.reload()
                batch = self.dataset.sample_batch(self.config.batch_size)
                metrics = self._train_step(batch)
                self.step += 1
                self.samples_processed += self.config.batch_size

                if self.step % self.config.log_every == 0:
                    self._log(metrics)

                if self.step % self.config.checkpoint.every_steps == 0:
                    self._save_checkpoint()
        except KeyboardInterrupt:
            self.console.log("Training interrupted by user.")
        finally:
            if self._wandb_run:
                wandb.finish()

    # ------------------------------------------------------------------ #
    def _train_step(self, batch: SelfPlayBatch) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)

        inputs = torch.from_numpy(batch.planes).to(self.device)
        targets_policy = torch.from_numpy(batch.policy_indices).to(self.device)
        targets_value = torch.from_numpy(batch.values).to(self.device)
        mask = torch.from_numpy(batch.mask).to(self.device)
        value_labels = torch.argmax(targets_value, dim=-1)

        with amp.autocast(device_type = self.device_type, dtype=self.amp_dtype):
            policy_logits, value_logits, value_q_logits = self.model(
                [inputs, None], return_all=True
            )
            B = policy_logits.shape[0]
            policy_logits = policy_logits.view(B, -1)
            value_logits = value_logits.view(B, 3)
            value_q_logits = value_q_logits.view(B, 3)

            policy_loss = F.cross_entropy(
                policy_logits, targets_policy, ignore_index=NULL_POLICY_INDEX
            )

            if mask.any():
                masked_value_logits = value_logits[mask]
                masked_labels = value_labels[mask]
                value_loss = F.cross_entropy(masked_value_logits, masked_labels)

                value_q_probs = torch.softmax(value_q_logits, dim=-1)
                value_q_loss = F.mse_loss(value_q_probs[mask], targets_value[mask])

                value_accuracy = (
                    (value_logits.argmax(dim=-1)[mask] == masked_labels)
                    .float()
                    .mean()
                )
            else:
                value_loss = value_logits.sum() * 0.0
                value_q_loss = value_logits.sum() * 0.0
                value_accuracy = torch.tensor(0.0, device=self.device)

            total_loss = policy_loss + value_loss + value_q_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        policy_accuracy = (
            (policy_logits.argmax(dim=-1) == targets_policy)
            .float()
            .mean()
            .item()
        )

        # Calculate policy entropy: -sum(p * log(p + eps))
        # Convert to float32 for accurate entropy calculation
        policy_logits_f32 = policy_logits.float()
        policy_probs = F.softmax(policy_logits_f32, dim=-1)
        eps = 1e-8
        policy_entropy = -(policy_probs * torch.log(policy_probs + eps)).sum(dim=-1).mean().item()

        metrics = {
            "loss/total": float(total_loss.detach().item()),
            "loss/policy": float(policy_loss.detach().item()),
            "loss/value": float(value_loss.detach().item()),
            "loss/value_q": float(value_q_loss.detach().item()),
            "acc/policy": policy_accuracy,
            "acc/value": float(value_accuracy.detach().item()),
            "entropy/policy": policy_entropy,
        }
        return metrics

    # ------------------------------------------------------------------ #
    def _log(self, metrics: Dict[str, float]) -> None:
        elapsed = time.time() - self.last_log_time
        if elapsed <= 0:
            elapsed = 1e-6
        positions_per_sec = (self.config.batch_size * self.config.log_every) / elapsed
        self.last_log_time = time.time()

        log_line = (
            f"[step {self.step}] "
            f"loss={metrics['loss/total']:.4f} "
            f"policy_acc={metrics['acc/policy']:.3f} "
            f"value_acc={metrics['acc/value']:.3f} "
            f"pps={positions_per_sec:,.0f}"
        )
        self.console.log(log_line)

        if self._wandb_run:
            wandb.log(
                {
                    **metrics,
                    "speed/positions_per_sec": positions_per_sec,
                    "global_step": self.step,
                }
            )

    # ------------------------------------------------------------------ #
    def _save_checkpoint(self) -> None:
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "config": self.config.to_dict(),
        }
        path = self.checkpoint_dir / f"step_{self.step:08d}.pt"
        torch.save(ckpt, path)
        self.console.log(f"Saved checkpoint to {path}")
        self._invoke_build_script(path)
        self._evaluate_checkpoint(path)
        self._prune_checkpoints()
        self.console.log("Waiting for half an hour before next evaluation...")
        time.sleep(1800) # half an hour

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step = int(ckpt.get("step", 0))
        self.console.log(f"Resumed from checkpoint {path} at step {self.step}")

    def _prune_checkpoints(self) -> None:
        keep = self.config.checkpoint.keep_last
        if keep <= 0:
            return
        files = sorted(self.checkpoint_dir.glob("step_*.pt"))
        for old in files[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass

    def _invoke_build_script(self, checkpoint_path: Path) -> None:
        script = self.config.checkpoint.build_script
        if script is None:
            return
        script_path = Path(script).resolve()
        if not script_path.is_file():
            self.console.log(f"[WARN] build script {script_path} not found; skipping TRT export.")
            return

        engine_dir = self.config.checkpoint.engine_dir
        iteration_file = self.config.checkpoint.iteration_file

        cmd = [
            script_path.as_posix(),
            checkpoint_path.resolve().as_posix(),
            str(self.step),
            engine_dir.as_posix(),
            iteration_file.as_posix(),
        ]
        try:
            subprocess.run(cmd, check=True)
            self.console.log(f"Updated TensorRT engine via {script_path}.")
        except Exception as exc:
            self.console.log(f"[WARN] build_trt.sh failed: {exc}")

    def _evaluate_checkpoint(self, checkpoint_path: Path) -> None:
        """Evaluate raw model against Stockfish and log ELO to wandb."""
        self.console.log(f"Evaluating checkpoint {checkpoint_path} against Stockfish (ELO {self._current_stockfish_elo})...")
        
        # Import play_move dynamically to avoid import path issues
        try:
            # Add project root to path if needed
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from play_move import main_model_vs_stockfish
        except ImportError as exc:
            self.console.log(f"[WARN] Failed to import play_move: {exc}")
            return
        
        # Load a fresh model instance from checkpoint
        eval_model = BT4().to(self.device)
        eval_model.eval()
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
            eval_model.load_state_dict(ckpt["model"])
        except Exception as exc:
            self.console.log(f"[WARN] Failed to load checkpoint for evaluation: {exc}")
            return
        
        # Run evaluation
        try:
            win_rate, estimated_elo = main_model_vs_stockfish(
                model=eval_model,
                model1_name="SelfPlay",
                device=self.device.type,
                num_games=100,
                temp=0.1,
                elo=self._current_stockfish_elo,
            )
            
            # Handle case where estimated_elo might be "N/A"
            elo_value = estimated_elo if isinstance(estimated_elo, (int, float)) else None
            
            # Log to wandb
            if self._wandb_run:
                log_dict = {
                    "eval/win_rate": win_rate,
                    "eval/stockfish_elo": self._current_stockfish_elo,
                    "global_step": self.step,
                }
                if elo_value is not None:
                    log_dict["eval/estimated_elo"] = elo_value
                wandb.log(log_dict)
            
            # Handle case where estimated_elo might be "N/A"
            if isinstance(estimated_elo, str):
                self.console.log(f"Evaluation complete: ELO=N/A, Win rate={win_rate:.3f}")
                # Don't update Stockfish ELO if we can't estimate
            else:
                self.console.log(f"Evaluation complete: ELO={estimated_elo:.1f}, Win rate={win_rate:.3f}")
                
                # Update Stockfish ELO for next evaluation
                self._current_stockfish_elo = max(int(estimated_elo), 1400)
                self.console.log(f"Next evaluation will use Stockfish ELO {self._current_stockfish_elo}")
            
        except Exception as exc:
            self.console.log(f"[WARN] Evaluation failed: {exc}")

