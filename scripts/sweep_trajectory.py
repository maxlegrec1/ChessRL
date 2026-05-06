"""Sweep every step_*.pt checkpoint in a run and produce trajectory plots.

For each checkpoint: load, run the 87-FEN learning_check fenset in one batch,
extract overall + per-theme metrics. Plot vs training step.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path("/mnt/2tb_2/ChessRL")
sys.path.insert(0, str(ROOT))

from src.eval.learning_check import (
    load_dataset,
    load_model,
    run_inference,
    best_legal_uci,
)


def step_from_path(p: Path) -> int:
    m = re.search(r"step_(\d+)", p.name)
    return int(m.group(1)) if m else -1


def evaluate_checkpoint(model, dataset, device):
    fens = [fen for _, fen, _ in dataset]
    p_logits, v_probs, vq_probs = run_inference(model, fens, device)
    by_theme = defaultdict(lambda: dict(n=0, vw_sum=0, vqw_sum=0, vargmax=0, vqargmax=0, mate=0, mate_n=0))
    overall = dict(n=0, vw_sum=0, vqw_sum=0, vargmax=0, vqargmax=0, mate=0, mate_n=0)
    for i, (theme, fen, mate_uci) in enumerate(dataset):
        v = v_probs[i]
        vq = vq_probs[i]
        for d in (by_theme[theme], overall):
            d["n"] += 1
            d["vw_sum"] += float(v[0])
            d["vqw_sum"] += float(vq[0])
            d["vargmax"] += int(np.argmax(v) == 0)
            d["vqargmax"] += int(np.argmax(vq) == 0)
        if mate_uci is not None:
            picked = best_legal_uci(p_logits[i], fen)
            for d in (by_theme[theme], overall):
                d["mate_n"] += 1
                d["mate"] += int(picked == mate_uci)
    return overall, dict(by_theme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="artifacts/runs/20260504_014403_e717")
    ap.add_argument("--cache", default=None, help="JSON cache to skip recomputed steps")
    ap.add_argument("--out-prefix", default="trajectory")
    ap.add_argument("--every", type=int, default=1, help="evaluate every Nth checkpoint")
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    ckpts = sorted(run_dir.glob("checkpoints/step_*.pt"), key=step_from_path)
    ckpts = [c for c in ckpts if step_from_path(c) > 0]
    ckpts = ckpts[::args.every]

    cache_path = Path(args.cache) if args.cache else (run_dir / "trajectory_cache.json")
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"loaded {len(cache)} cached steps from {cache_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset(seed=42)
    print(f"dataset: {len(dataset)} fens; checkpoints: {len(ckpts)}")

    for c in ckpts:
        step = step_from_path(c)
        key = str(step)
        if key in cache:
            continue
        try:
            model = load_model(c, device)
            overall, by_theme = evaluate_checkpoint(model, dataset, device)
            cache[key] = {"overall": overall, "themes": by_theme}
            del model
            torch.cuda.empty_cache()
            print(f"step={step:6d}  V[W]={overall['vw_sum']/overall['n']:.3f}  "
                  f"Vq[W]={overall['vqw_sum']/overall['n']:.3f}  "
                  f"mate={overall['mate']}/{overall['mate_n']}")
        except Exception as e:
            print(f"step={step}: ERROR {e}")
            continue
        # save incrementally
        cache_path.write_text(json.dumps(cache))

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(int(k) for k in cache.keys())
    def col(field, source="overall", theme=None):
        out = []
        for s in steps:
            d = cache[str(s)]["overall"] if theme is None else cache[str(s)]["themes"].get(theme)
            if d is None or d.get("n", 0) == 0:
                out.append(np.nan)
            else:
                if field == "v_w":
                    out.append(d["vw_sum"] / d["n"])
                elif field == "vq_w":
                    out.append(d["vqw_sum"] / d["n"])
                elif field == "v_argmax":
                    out.append(100 * d["vargmax"] / d["n"])
                elif field == "vq_argmax":
                    out.append(100 * d["vqargmax"] / d["n"])
                elif field == "mate_pct":
                    out.append(100 * d["mate"] / max(1, d["mate_n"]))
        return np.array(out)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    ax = axes[0, 0]
    ax.plot(steps, col("v_w"), label="V[W]", lw=1.5)
    ax.plot(steps, col("vq_w"), label="Vq[W]", lw=1.5)
    ax.axhline(0.33, color="gray", lw=0.5, ls="--")
    ax.set_title("Overall mean P(Win)")
    ax.set_ylabel("probability")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, col("v_argmax"), label="V→W %", lw=1.5)
    ax.plot(steps, col("vq_argmax"), label="Vq→W %", lw=1.5)
    ax.set_title("Argmax-is-Win rate")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for theme in ["mate_1_back_rank", "mate_1_queen_back", "mate_1_two_rooks"]:
        ax.plot(steps, col("mate_pct", theme=theme), label=theme, lw=1.2)
    ax.plot(steps, col("mate_pct"), color="black", lw=1.5, label="overall")
    ax.set_title("Mate-in-1 found by policy argmax (%)")
    ax.set_ylabel("%")
    ax.set_xlabel("training step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for theme in ["end_kqk", "end_kqqk", "end_kqrk", "end_krk", "end_krrk", "end_kbnk"]:
        ax.plot(steps, col("v_w", theme=theme), label=theme, lw=1.0)
    ax.set_title("V[W] by endgame theme")
    ax.set_ylabel("probability")
    ax.set_xlabel("training step")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Learning trajectory — {run_dir.name} (steps {steps[0]} → {steps[-1]})")
    fig.tight_layout()
    out = run_dir / f"{args.out_prefix}.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
