"""Re-plot the elo chain with the correct sign convention.

CSV's delta_elo = (older - newer). For a chain in time order, the per-interval
*gain* of the newer model is therefore -delta_elo, and cumulative Elo of model
at step s = -sum(delta_elo) up to that step.

Merges multiple CSVs (e.g., the original 5k→115k + the 115k→155k extension).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSVS = [
    ROOT / "artifacts/exp/elo_chain/results.csv",       # 5k→115k (original)
    ROOT / "artifacts/exp/elo_chain_ext/results.csv",   # 115k→155k extension
    ROOT / "artifacts/exp/elo_chain_ext2/results.csv",  # 155k→190k extension
    ROOT / "artifacts/exp/elo_chain_ext3/results.csv",  # 190k→220k extension
]
OUT = ROOT / "artifacts/exp/elo_chain/elo_chain_fixed.png"

# Policy-change markers placed by training step.
MARKERS = [
    (63500,  "mirror-bug fix + restart",              "orange"),
    (140000, "dirichlet bump α 0.12→0.3, ε 0.05→0.25", "purple"),
]

# Collect + de-duplicate rows across CSVs (in case overlap exists).
seen = set()
rows = []
for path in CSVS:
    if not path.is_file():
        continue
    for r in csv.DictReader(path.open()):
        key = (int(r["from_step"]), int(r["to_step"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
rows.sort(key=lambda r: int(r["from_step"]))

labels = [f"{int(r['from_step'])//1000}k→{int(r['to_step'])//1000}k" for r in rows]
gains  = [-float(r["delta_elo"]) for r in rows]   # newer - older (positive = improvement)

cum_steps = [int(rows[0]["from_step"])]
cum = [0.0]
running = 0.0
for r in rows:
    running += -float(r["delta_elo"])
    cum_steps.append(int(r["to_step"]))
    cum.append(running)

fig, axes = plt.subplots(2, 1, figsize=(15, 9))

bars = axes[0].bar(labels, gains,
                   color=["seagreen" if g > 0 else "lightcoral" for g in gains],
                   edgecolor="black", linewidth=0.5)
for bar, g in zip(bars, gains):
    axes[0].text(bar.get_x() + bar.get_width()/2, g + (5 if g >= 0 else -5),
                 f"{g:+.0f}", ha="center", va="bottom" if g >= 0 else "top",
                 fontsize=7)
axes[0].set_ylabel("ΔElo per 5k-step interval  (newer − older)")
axes[0].set_title("Per-interval Elo gain  (50 games/pair, MCTS-800 width=1 temp_root=0, "
                  "8 random opening plies)")
axes[0].axhline(0, color="grey", lw=1)
axes[0].grid(alpha=.3, axis="y")
plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=7)

axes[1].plot(cum_steps, cum, marker="o", color="darkblue", lw=2, ms=5)
axes[1].fill_between(cum_steps, cum, 0, alpha=.15, color="darkblue")
for step, label, color in MARKERS:
    if cum_steps[0] <= step <= cum_steps[-1]:
        axes[1].axvline(step, color=color, ls="--", lw=2,
                        label=f"{label}  (~step {step//1000}k)")
axes[1].set_xlabel("training step")
axes[1].set_ylabel(f"cumulative Elo (relative to {cum_steps[0]//1000}k)")
axes[1].set_title(f"Cumulative Elo trajectory   "
                  f"(final: {cum[-1]:+.0f} Elo from {cum_steps[0]//1000}k → {cum_steps[-1]//1000}k)")
axes[1].grid(alpha=.3)
axes[1].axhline(0, color="grey", lw=1)
axes[1].legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches="tight")

# Concise text summary
print("Per-interval gain (newer - older):")
for lbl, g in zip(labels, gains):
    marker = "★" if g >= 100 else (" " if g >= 0 else "·")
    print(f"  {marker} {lbl:>12}  {g:+6.0f}")
print(f"\nTotal cumulative gain {cum_steps[0]//1000}k → {cum_steps[-1]//1000}k: "
      f"{cum[-1]:+.0f} Elo")

# Segment splits keyed on the markers.
def seg_sum(lo, hi):
    return sum(g for lbl, g in zip(labels, gains)
               if lo <= int(lbl.split('k→')[0])*1000 < hi)
print("\nSegment Elo gains:")
edges = [cum_steps[0]] + [m[0] for m in MARKERS if cum_steps[0] < m[0] < cum_steps[-1]] + [cum_steps[-1]]
for a, b in zip(edges[:-1], edges[1:]):
    g = seg_sum(a, b)
    n_intervals = sum(1 for lbl in labels if a <= int(lbl.split('k→')[0])*1000 < b)
    print(f"  {a//1000:>3}k → {b//1000:>3}k  ({n_intervals:>2} pairs):  {g:+6.0f} Elo  "
          f"({(g/(b-a)*10000) if b>a else 0:+5.1f} Elo per 10k steps)")
print(f"\nplot: {OUT}")
