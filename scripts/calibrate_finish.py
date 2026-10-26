"""Finish an anchor calibration from its log: replay only the missing pairs, then fit.

ladder.calibrate() keeps its game results in memory and prints one line per finished pair
("a vs b: S/N"). If a worker hangs (a crashed Stockfish once left one waiting forever), the
finished pairs are still in the log: a pair's total score is all the Bradley-Terry fit needs,
so this replays the missing pairs and writes the same ratings file calibrate() would.

  uv run python scripts/calibrate_finish.py --log calib_sf.log --openings sf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rating import ladder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--openings", choices=("sf", "random"), default="sf")
    p.add_argument("--games-per-pair", type=int, default=100)
    a = p.parse_args()

    names = list(ladder.ANCHORS)
    pairs = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
    pairs += [(names[i], names[i + 2]) for i in range(len(names) - 2)]
    totals: dict[tuple[str, str], tuple[float, int]] = {}
    for line in a.log.read_text().splitlines():
        m = re.match(r"(\S+) vs (\S+): ([\d.]+)/(\d+)", line.strip())
        if m:
            totals[(m[1], m[2])] = (float(m[3]), int(m[4]))
    openings = ladder.load_openings(a.openings, a.games_per_pair // 2, seed=99)
    for pair in pairs:
        if pair not in totals:
            out = ladder._play_pair(*pair, openings)
            totals[pair] = (sum(s for *_, s in out), len(out))
    # Expand each pair's total into games with that total (wins, at most one draw, losses).
    results = []
    for (x, y), (score, n) in totals.items():
        wins, half = int(score), score - int(score) >= 0.5
        games = [1.0] * wins + ([0.5] if half else [])
        results += [(x, y, s) for s in games + [0.0] * (n - len(games))]
    ratings = {n: 200.0 * i for i, n in enumerate(names)}
    for _ in range(200):
        for n in names[1:]:
            vs = [(ratings[b], s) for x, b, s in results if x == n] + [(ratings[x], 1 - s) for x, b, s in results if b == n]
            ratings[n] = ladder.fit_rating(vs, prior=ratings[n])[0]
    ratings = {n: round(r - ratings["random"], 1) for n, r in ratings.items()}
    out = ladder.ANCHOR_RATINGS_FILES[a.openings]
    out.write_text(json.dumps({"ratings": ratings, "games_per_pair": a.games_per_pair, "openings": a.openings,
                               "pairs": {f"{x} vs {y}": f"{s}/{n}" for (x, y), (s, n) in totals.items()}}, indent=2))
    print(json.dumps(ratings, indent=2))


if __name__ == "__main__":
    main()
