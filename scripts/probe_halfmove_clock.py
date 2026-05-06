"""Probe the value head's sensitivity to the half-move clock.

For each test position, sweep halfmove_clock from 0 to 99 (the 50-move rule
fires at 100), patching only that field of the FEN. Run the model and record
V[W] and Vq[W] at each clock value. If the model has learned that approaching
the 50-move rule reduces winning chances, V[W] should monotonically decrease
toward 0.33 (uniform-ish) as clock → 100.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import chess
import numpy as np
import torch

ROOT = Path("/mnt/2tb_2/ChessRL")
sys.path.insert(0, str(ROOT))

from src.models.small.fen_encoder import fen_to_tensor
from src.models.small.small_model import SmallModel


# Positions that the latest learning_check shows as cleanly winning (V[W]>0.5)
TEST_POSITIONS = [
    # (label, fen with halfmove=0)
    ("KQQ_v_k",      "8/8/8/8/8/4k3/Q7/QK6 w - - 0 1"),
    ("KQR_v_k",      "8/8/8/8/8/4k3/Q7/RK6 w - - 0 1"),
    ("KRR_v_k",      "8/8/8/8/8/4k3/R7/RK6 w - - 0 1"),
    ("KQ_v_k",       "8/8/8/8/8/4k3/8/QK6 w - - 0 1"),
    ("KR_v_k",       "8/8/8/8/8/4k3/8/RK6 w - - 0 1"),
    ("KBN_v_k",      "8/8/8/8/8/4k3/8/BNK5 w - - 0 1"),
    ("mate_back_rank","6k1/5ppp/8/8/8/8/8/3Q3K w - - 0 1"),
    ("mate_q_back",  "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"),
    ("startpos",     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
]


def patch_halfmove(fen: str, halfmove: int) -> str:
    parts = fen.split()
    # FEN: pieces stm castles ep half full
    if len(parts) < 6:
        # default fullmove if missing
        parts = parts + ["-"] * (6 - len(parts))
    parts[4] = str(halfmove)
    return " ".join(parts)


@torch.no_grad()
def evaluate(model, fens, device):
    arr = np.stack([fen_to_tensor(f) for f in fens]).astype(np.float32)
    x = torch.from_numpy(arr).to(device)
    p, v, vq, mlh = model(x, return_all=True)
    v = torch.softmax(v.float(), dim=-1).cpu().numpy()
    vq = torch.softmax(vq.float(), dim=-1).cpu().numpy()
    return v, vq, mlh.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--clocks", default="0,10,25,50,75,90,99",
                    help="comma-separated halfmove values to probe")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        run_dir = ROOT / "artifacts/runs/20260504_014403_e717"
        ckpts = sorted(run_dir.glob("checkpoints/step_*.pt"))
        ckpt_path = ckpts[-1]
    print(f"checkpoint: {ckpt_path.name}")

    model = SmallModel().to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)

    clocks = [int(x) for x in args.clocks.split(",")]

    # one big batch: positions × clocks
    fens = []
    labels = []
    pclocks = []
    for label, base_fen in TEST_POSITIONS:
        for clk in clocks:
            fens.append(patch_halfmove(base_fen, clk))
            labels.append(label)
            pclocks.append(clk)

    v, vq, mlh = evaluate(model, fens, device)

    # Print per-position table
    print()
    print(f"{'position':<20} {'clock':>5} {'V[W]':>7} {'V[D]':>7} {'V[L]':>7}  {'Vq[W]':>7} {'Vq[D]':>7} {'Vq[L]':>7}  {'mlh':>5}")
    print("-" * 100)
    cur_label = None
    for i, label in enumerate(labels):
        if label != cur_label:
            if cur_label is not None:
                print()
            cur_label = label
        print(f"{label:<20} {pclocks[i]:>5} "
              f"{v[i,0]:>7.3f} {v[i,1]:>7.3f} {v[i,2]:>7.3f}  "
              f"{vq[i,0]:>7.3f} {vq[i,1]:>7.3f} {vq[i,2]:>7.3f}  "
              f"{mlh[i]:>5.3f}")

    # Summary: per position, how much does V[W] drop from clock=0 to clock=99?
    print()
    print("=== Summary: V[W] sensitivity to half-move clock ===")
    print(f"{'position':<20} {'V[W]@0':>8} {'V[W]@99':>8} {'ΔV[W]':>8}    {'Vq[W]@0':>8} {'Vq[W]@99':>9} {'ΔVq[W]':>8}    {'mlh@0':>6} {'mlh@99':>7} {'Δmlh':>6}")
    print("-" * 110)
    n_clocks = len(clocks)
    for pi in range(len(TEST_POSITIONS)):
        label = TEST_POSITIONS[pi][0]
        i0 = pi * n_clocks                # clock=0 row
        i99 = pi * n_clocks + n_clocks - 1  # clock=99 row
        dvw = v[i99, 0] - v[i0, 0]
        dvqw = vq[i99, 0] - vq[i0, 0]
        dmlh = float(mlh[i99] - mlh[i0])
        print(f"{label:<20} {v[i0,0]:>8.3f} {v[i99,0]:>8.3f} {dvw:>+8.3f}    "
              f"{vq[i0,0]:>8.3f} {vq[i99,0]:>9.3f} {dvqw:>+8.3f}    "
              f"{float(mlh[i0]):>6.3f} {float(mlh[i99]):>7.3f} {dmlh:>+6.3f}")


if __name__ == "__main__":
    main()
