"""Plot autobrawl daemon win-rate trend across checkpoints."""

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts/exp/sf1400_autobrawl/state.json"
OUT = ROOT / "artifacts/exp/sf1400_autobrawl/winrate_trend.png"

d = json.loads(STATE.read_text())
results = d.get("results", {})

rows = []
for s in sorted(int(x) for x in results):
    line = results[str(s)]
    m = re.search(r"(\d+)W / (\d+)D / (\d+)L\s*->\s*(\d+\.\d+)/(\d+)\s*\(([\d\.]+)%\)", line)
    if not m:
        continue
    w, dd, l, score, n, pct = m.groups()
    w, dd, l, n = int(w), int(dd), int(l), int(n)
    pct = float(pct)
    p = pct / 100
    elo = (
        1400 + (-400 * math.log10((1 - p) / p))
        if 0 < p < 1
        else (1400 + 800 if p == 1 else 1400 - 800)
    )
    rows.append((s, pct, elo, w, dd, l, n))

steps = [r[0] for r in rows]
pcts = [r[1] for r in rows]
elos = [r[2] for r in rows]


# 3-pair rolling mean for smoothing
def rolling(xs, k=3):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k // 2)
        hi = min(len(xs), i + k // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


pcts_smooth = rolling(pcts, 3)

fig, ax1 = plt.subplots(figsize=(13, 6))

# Primary: win rate
ax1.scatter(steps, pcts, marker="o", s=40, color="steelblue", label="brawl % (n=50)")
ax1.plot(steps, pcts_smooth, color="darkblue", lw=2.5, label="3-brawl rolling mean")
ax1.axhline(50, color="red", ls="--", lw=1.5, alpha=0.7, label="parity with SF-1400")
ax1.set_xlabel("training step")
ax1.set_ylabel("win % vs Stockfish-16 @ UCI_Elo=1400 (sequential, no rand)", color="darkblue")
ax1.tick_params(axis="y", labelcolor="darkblue")
ax1.grid(alpha=0.3)
ax1.set_ylim(25, 65)
ax1.set_xlim(min(steps) - 2000, max(steps) + 2000)

# Annotate min/max
i_min = pcts.index(min(pcts))
i_max = pcts.index(max(pcts))
ax1.annotate(
    f"  trough: {pcts[i_min]:.0f}%",
    xy=(steps[i_min], pcts[i_min]),
    xytext=(steps[i_min] + 3000, pcts[i_min] - 2.5),
    fontsize=9,
    color="firebrick",
)
ax1.annotate(
    f"  best: {pcts[i_max]:.0f}%",
    xy=(steps[i_max], pcts[i_max]),
    xytext=(steps[i_max] - 7000, pcts[i_max] + 2),
    fontsize=9,
    color="darkgreen",
)

# Secondary: implied Elo
ax2 = ax1.twinx()
ax2.plot(steps, elos, alpha=0)  # invisible, just for axis sync
ax2.set_ylabel("implied model Elo (SF-16 anchor)", color="grey")
ax2.tick_params(axis="y", labelcolor="grey")

# Match Elo range to pct range
import numpy as np

pcts_arr = np.linspace(25, 65, 5)
elo_arr = [1400 + (-400 * math.log10((1 - p / 100) / (p / 100))) for p in pcts_arr]
ax2.set_yticks(elo_arr)
ax2.set_yticklabels([f"{e:.0f}" for e in elo_arr])
ax2.set_ylim(elo_arr[0], elo_arr[-1])

plt.title(
    f"Autobrawl daemon win-rate trajectory  "
    f"({len(rows)} brawls, n=50 each, SF-16 UCI_Elo=1400, 100ms, no rand openings)\n"
    f"Step {min(steps) // 1000}k → {max(steps) // 1000}k:  "
    f"{pcts[0]:.0f}% → {pcts[-1]:.0f}%  "
    f"(Elo ~{elos[0]:.0f} → ~{elos[-1]:.0f}, Δ≈{elos[-1] - elos[0]:+.0f})"
)
ax1.legend(loc="upper left")
plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"plot: {OUT}")
print(
    f"summary: {len(rows)} brawls, "
    f"step {min(steps) // 1000}k→{max(steps) // 1000}k, "
    f"win% {pcts[0]:.0f}→{pcts[-1]:.0f}, Elo {elos[0]:.0f}→{elos[-1]:.0f} (Δ{elos[-1] - elos[0]:+.0f})"
)
