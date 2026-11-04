"""Follow-up analyses of a search_quality.py --dump (per-position targets and moves).

1. Equal sharpness. A sharper target scores a lower sampled loss whatever its quality, so
   each PUCT target set is sharpened (p^(1/T), T found by bisection) until its effective
   number of moves matches a Gumbel setting's, and the sampled losses are compared.
2. Tree reuse. The share of PUCT's root visits under the move it plays (what a reused tree
   would keep for the next search), and how often that subtree alone holds 50 visits (the
   size of a fast search).

  uv run python scripts/search_quality_report.py --dump rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_POSITIONS = Path.home() / "chessrl-runs/runC-open0/gap_positions.json"


def score(
    targets: dict[int, dict[str, float]], positions: list[dict]
) -> tuple[float, float, float]:
    """Mean sampled loss, share of sampled moves losing > 0.10, effective moves."""
    loss, blunder, ent = [], [], []
    for i, t in targets.items():
        s = positions[i]["scores"]
        best = max(s.values())
        tot = sum(p for m, p in t.items() if m in s) or 1.0
        probs = {m: p / tot for m, p in t.items() if m in s}
        loss.append(sum(p * (best - s[m]) for m, p in probs.items()))
        blunder.append(sum(p for m, p in probs.items() if best - s[m] > 0.10))
        ent.append(-sum(p * math.log(p) for p in probs.values() if p > 0))
    n = len(loss)
    return sum(loss) / n, sum(blunder) / n, math.exp(sum(ent) / n)


def sharpen(t: dict[str, float], temp: float) -> dict[str, float]:
    w = {m: p ** (1.0 / temp) for m, p in t.items() if p > 0}
    s = sum(w.values())
    return {m: v / s for m, v in w.items()}


def matched(rows: dict, eff_target: float, positions: list[dict]) -> tuple[float, float]:
    """(temperature, sampled loss) of the PUCT targets sharpened to eff_target effective moves."""
    lo, hi = 0.1, 3.0  # higher T = flatter = more effective moves
    for _ in range(40):
        mid = (lo + hi) / 2
        _, _, eff = score({i: sharpen(r["target"], mid) for i, r in rows.items()}, positions)
        lo, hi = (mid, hi) if eff < eff_target else (lo, mid)
    t = (lo + hi) / 2
    return t, score({i: sharpen(r["target"], t) for i, r in rows.items()}, positions)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    a = ap.parse_args()
    positions = json.loads(a.positions.read_text())
    by: dict[tuple[int, str], dict[int, dict]] = defaultdict(dict)
    for line in a.dump.read_text().splitlines():
        r = json.loads(line)
        by[(r["step"], r["config"])][r["index"]] = r
    steps = sorted({s for s, _ in by})
    puct = [c for c in ("puct:50", "puct:100", "puct:200", "puct:400") if (steps[0], c) in by]
    gumbel = sorted((c for s, c in by if s == steps[0] and c.startswith("gumbel:")),
                    key=lambda c: int(c.split(":")[1]))  # fmt: skip

    for step in steps:
        print(f"\nstep {step}: sampled loss at the same effective number of moves")
        print(
            f"{'setting':<11} {'evals':>5} {'eff':>5} {'loss':>7}"
            + "".join(f"{p:>16}" for p in puct)
        )
        for g in gumbel:
            rows = by[(step, g)]
            loss, _, eff = score({i: r["target"] for i, r in rows.items()}, positions)
            evals = sum(r["evals"] for r in rows.values()) / len(rows)
            cells = [matched(by[(step, p)], eff, positions) for p in puct]
            print(
                f"{g:<11} {evals:>5.0f} {eff:>5.2f} {loss:>7.4f}"
                + "".join(f"{f'{pl:.4f} T{t:.2f}':>16}" for t, pl in cells)
            )
        print(f"step {step}: tree reuse (share of the root visits under the move played)")
        for config in ("puct:400", "puctnoise:400", "puct:50"):
            rows = by.get((step, config))
            if not rows:
                continue
            sims = int(config.split(":")[1])
            shares = sorted(r["target"][r["played"]] for r in rows.values())
            covers = sum(sh * sims >= 50 for sh in shares) / len(shares)
            print(
                f"  {config:<14} mean {sum(shares) / len(shares):.2f}, median "
                f"{shares[len(shares) // 2]:.2f}; >= 50 visits in {covers:.0%} of positions"
            )


if __name__ == "__main__":
    main()
