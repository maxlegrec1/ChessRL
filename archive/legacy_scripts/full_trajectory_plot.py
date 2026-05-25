"""Render the 'full' multi-panel trajectory plot from the cache that
sweep_trajectory.py produces.

Six panels:
  1. Overall V[W] / Vq[W] (smoothed)
  2. Overall V→W% / Vq→W% (smoothed)
  3. V[W] per theme (smoothed)
  4. Vq[W] per theme (smoothed)
  5. Mate-in-1 found (per theme + overall)
  6. V→W % per theme (smoothed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/2tb_2/ChessRL")
RUN_DIR = ROOT / "artifacts/runs/20260504_014403_e717"
CACHE = RUN_DIR / "trajectory_cache.json"
OUT = RUN_DIR / "full_trajectory.png"

THEMES = [
    "end_kqqk", "end_kqrk", "end_krrk",
    "end_kqk", "end_krk", "end_kbnk",
    "mate_1_back_rank", "mate_1_queen_back", "mate_1_two_rooks",
]


def smooth(arr, w=5):
    arr = np.asarray(arr, dtype=float)
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def col(cache, steps, field, theme=None):
    out = []
    for s in steps:
        d = cache[str(s)]["overall"] if theme is None else cache[str(s)]["themes"].get(theme)
        if d is None or d.get("n", 0) == 0:
            out.append(np.nan)
            continue
        n = d["n"]
        mn = d.get("mate_n", 0) or 1
        if field == "v_w":
            out.append(d["vw_sum"] / n)
        elif field == "vq_w":
            out.append(d["vqw_sum"] / n)
        elif field == "v_argmax":
            out.append(100 * d["vargmax"] / n)
        elif field == "vq_argmax":
            out.append(100 * d["vqargmax"] / n)
        elif field == "mate_pct":
            out.append(100 * d["mate"] / mn)
    return np.array(out)


def main():
    if not CACHE.exists():
        print(f"no cache at {CACHE}; run scripts/sweep_trajectory.py first", file=sys.stderr)
        return 2
    cache = json.loads(CACHE.read_text())
    steps = sorted(int(k) for k in cache.keys())
    print(f"loaded {len(steps)} checkpoints from {steps[0]} to {steps[-1]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

    # MLH integration step (first checkpoint with the new arch)
    MLH_STEP = 11900

    # 1. Overall V[W] / Vq[W]
    ax = axes[0, 0]
    vw = col(cache, steps, "v_w")
    vqw = col(cache, steps, "vq_w")
    ax.plot(steps, vw, color="C0", alpha=0.25, lw=1)
    ax.plot(steps, vqw, color="C1", alpha=0.25, lw=1)
    ax.plot(steps, smooth(vw, 5), color="C0", lw=2, label="V[W] (smoothed)")
    ax.plot(steps, smooth(vqw, 5), color="C1", lw=2, label="Vq[W] (smoothed)")
    ax.axvline(MLH_STEP, color="purple", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("Overall mean P(side-to-move wins)")
    ax.set_ylabel("probability")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Argmax-is-Win %
    ax = axes[0, 1]
    va = col(cache, steps, "v_argmax")
    vqa = col(cache, steps, "vq_argmax")
    ax.plot(steps, va, color="C0", alpha=0.25, lw=1)
    ax.plot(steps, vqa, color="C1", alpha=0.25, lw=1)
    ax.plot(steps, smooth(va, 5), color="C0", lw=2, label="V→W % (smoothed)")
    ax.plot(steps, smooth(vqa, 5), color="C1", lw=2, label="Vq→W % (smoothed)")
    ax.axvline(MLH_STEP, color="purple", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("% positions where argmax = W")
    ax.set_ylabel("%")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # 3. V[W] per theme
    ax = axes[1, 0]
    for i, theme in enumerate(THEMES):
        y = col(cache, steps, "v_w", theme=theme)
        ax.plot(steps, smooth(y, 5), lw=1.4, label=theme)
    ax.axvline(MLH_STEP, color="purple", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("V[W] per theme (5-pt smoothed)")
    ax.set_ylabel("V[W]")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    # 4. Vq[W] per theme
    ax = axes[1, 1]
    for i, theme in enumerate(THEMES):
        y = col(cache, steps, "vq_w", theme=theme)
        ax.plot(steps, smooth(y, 5), lw=1.4, label=theme)
    ax.axvline(MLH_STEP, color="purple", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("Vq[W] per theme (5-pt smoothed)")
    ax.set_ylabel("Vq[W]")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    # 5. Mate-in-1 by theme + overall
    ax = axes[2, 0]
    for theme in ["mate_1_back_rank", "mate_1_queen_back", "mate_1_two_rooks"]:
        y = col(cache, steps, "mate_pct", theme=theme)
        ax.plot(steps, y, lw=1.0, label=theme, alpha=0.7)
    ax.plot(steps, col(cache, steps, "mate_pct"), color="black", lw=1.5, label="overall")
    ax.axvline(MLH_STEP, color="purple", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("Mate-in-1 found by policy argmax (% positions, 0/N visible as 0)")
    ax.set_ylabel("%")
    ax.set_xlabel("training step")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # 6. V→W % per theme
    ax = axes[2, 1]
    for theme in THEMES:
        y = col(cache, steps, "v_argmax", theme=theme)
        ax.plot(steps, smooth(y, 5), lw=1.2, label=theme)
    ax.axvline(MLH_STEP, color="purple", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("V→W % per theme (5-pt smoothed)")
    ax.set_ylabel("%")
    ax.set_xlabel("training step")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Run 20260504_014403_e717 — full trajectory ({len(steps)} ckpts, "
        f"steps {steps[0]} → {steps[-1]}; purple line = MLH integration at step {MLH_STEP})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=110)
    print(f"saved {OUT}")


if __name__ == "__main__":
    sys.exit(main())
