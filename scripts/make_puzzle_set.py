"""Build the fixed eval puzzle set from the Lichess puzzle database (CC0).

Takes the first `--per-bin` puzzles (file order, so the set is deterministic)
in each rating band that are well established (Popularity >= 80, NbPlays >= 500).

  uv run python scripts/make_puzzle_set.py /path/lichess_db_puzzle.csv data/eval/puzzles_1k.csv
"""

from __future__ import annotations

import argparse
import csv

BANDS = [(0, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 10_000)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--per-bin", type=int, default=200)
    a = p.parse_args()
    picked: dict[int, list[dict]] = {i: [] for i in range(len(BANDS))}
    with open(a.src, newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["Popularity"]) < 80 or int(row["NbPlays"]) < 500:
                continue
            rating = int(row["Rating"])
            band = next(i for i, (lo, hi) in enumerate(BANDS) if lo <= rating < hi)
            if len(picked[band]) < a.per_bin:
                picked[band].append(row)
            if all(len(v) >= a.per_bin for v in picked.values()):
                break
    with open(a.dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["PuzzleId", "FEN", "Moves", "Rating", "Themes"])
        for band in picked.values():
            for row in band:
                w.writerow(
                    [row["PuzzleId"], row["FEN"], row["Moves"], row["Rating"], row["Themes"]]
                )
    print({f"{lo}-{hi}": len(picked[i]) for i, (lo, hi) in enumerate(BANDS)})


if __name__ == "__main__":
    main()
