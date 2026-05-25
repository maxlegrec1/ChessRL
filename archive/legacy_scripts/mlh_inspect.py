"""Quick MLH diagnostic.

(1) Scan the active buffer:
      - fraction of rows with plies_remaining != -1 (= valid MLH targets)
      - histogram of plies_remaining for valid rows
(2) Sample N valid rows, run the latest student through forward, compare
    predicted MLH (sigmoid in [0,1]) vs target tanh(plies/25).
"""
from __future__ import annotations

import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/2tb_2/ChessRL")
sys.path.insert(0, str(ROOT))

from src.models.small.fen_encoder import fen_to_tensor
from src.models.small.small_model import SmallModel

RECORD_SIZE = 148
PLIES_OFFSET = 140
FEN_OFFSET = 0
POLICY_OFFSET = 100


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("checkpoints/step_*.pt"))
    return ckpts[-1]


def main():
    run_dir = ROOT / "artifacts/runs/20260504_014403_e717"
    buf_path = run_dir / "buffer/buffer.bin"

    raw = buf_path.read_bytes()
    n_records = len(raw) // RECORD_SIZE
    print(f"buffer: {n_records:,} records ({len(raw)/1e6:.1f} MB)")

    plies_arr = np.frombuffer(
        np.frombuffer(raw, dtype=np.uint8).reshape(n_records, RECORD_SIZE)[:, PLIES_OFFSET:PLIES_OFFSET+4].copy(),
        dtype=np.float32,
    )
    valid = plies_arr != -1.0
    n_valid = int(valid.sum())
    print(f"valid (plies != -1): {n_valid:,} / {n_records:,} ({100*n_valid/n_records:.2f}%)")

    if n_valid > 0:
        v = plies_arr[valid]
        bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        prev = -float("inf")
        print("\nplies_remaining distribution (valid rows):")
        print(f"  min={v.min():.0f}  max={v.max():.0f}  mean={v.mean():.1f}  median={np.median(v):.1f}")
        for b in bins:
            count = int(((v > prev) & (v <= b)).sum())
            pct = 100 * count / n_valid
            print(f"    ({prev:>5.0f}, {b:>5}] : {count:>10,}  {pct:5.1f}%")
            prev = b
        count_above = int((v > prev).sum())
        if count_above:
            print(f"    ({prev:>5.0f}, +inf): {count_above:>10,}  {100*count_above/n_valid:5.1f}%")

    if n_valid == 0:
        print("\nno valid rows yet — MLH targets not populated. Skipping forward-pass check.")
        return

    # Sample up to 4096 valid rows and run forward
    n_sample = min(4096, n_valid)
    valid_idx = np.where(valid)[0]
    rng = np.random.default_rng(42)
    pick = rng.choice(valid_idx, size=n_sample, replace=False)

    # Pull FENs
    print(f"\nsampling {n_sample} valid rows for forward pass...")
    fens = []
    plies_pick = []
    for i in pick:
        fen_bytes = raw[i*RECORD_SIZE + FEN_OFFSET : i*RECORD_SIZE + POLICY_OFFSET]
        fen = fen_bytes.split(b"\0", 1)[0].decode("utf-8", errors="ignore")
        if fen:
            fens.append(fen)
            plies_pick.append(plies_arr[i])

    plies_pick = np.array(plies_pick, dtype=np.float32)
    # MLH head now predicts raw plies/25 (linear, no activation). Target is
    # the same linear transform of the buffer's plies-remaining, clamped at
    # the trainer's saturation cap.
    target_mlh = np.clip(np.maximum(plies_pick, 0) / 25.0, 0.0, 4.0)

    ckpt_path = latest_checkpoint(run_dir)
    print(f"loading {ckpt_path.name}")
    model = SmallModel().cuda().eval()
    state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"], strict=False)

    # Encode in chunks
    BATCH = 512
    preds = []
    with torch.no_grad():
        for off in range(0, len(fens), BATCH):
            chunk = fens[off:off+BATCH]
            x = torch.from_numpy(np.stack([fen_to_tensor(f) for f in chunk]).astype(np.float32)).cuda()
            _, _, _, mlh, _ = model(x, return_all=True)
            preds.append(mlh.cpu().numpy())
    preds = np.concatenate(preds)

    err = preds - target_mlh
    abs_err = np.abs(err)
    print(f"\nMLH head vs target on {len(fens):,} valid buffer rows:")
    print(f"  pred mean={preds.mean():.3f}  target mean={target_mlh.mean():.3f}")
    print(f"  MAE = {abs_err.mean():.4f}    RMSE = {np.sqrt((err**2).mean()):.4f}")
    print(f"  pred range = [{preds.min():.3f}, {preds.max():.3f}]")

    # Bucketed accuracy: bin target by plies-remaining range, show pred vs target
    print("\nper-bucket pred vs target (mean):")
    bins = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 1000)]
    for lo, hi in bins:
        mask = (plies_pick > lo) & (plies_pick <= hi)
        if mask.sum() == 0:
            continue
        print(f"  plies in ({lo:>3}, {hi:>3}]:  N={int(mask.sum()):>4}  "
              f"target_mean={target_mlh[mask].mean():.3f}  "
              f"pred_mean={preds[mask].mean():.3f}  "
              f"MAE={abs_err[mask].mean():.4f}")


if __name__ == "__main__":
    main()
