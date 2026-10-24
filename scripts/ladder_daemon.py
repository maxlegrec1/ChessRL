"""Live strength tracking for one run: every N minutes, rate the newest engine.

Each rating point = puzzle accuracy (1k Lichess puzzles, MCTS) + games against
the two calibrated anchors nearest the previous estimate. Results append to
<run>/ladder/ladder.jsonl. Every run gets the same cadence so the eval load is
identical across A/B arms.

  uv run python scripts/ladder_daemon.py --run-dir artifacts/runs/<id>
  uv run python scripts/ladder_daemon.py --calibrate      # one-off anchor round robin
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rating import ladder  # noqa: E402


def latest_engine(run_dir: Path) -> tuple[int, str] | None:
    it = run_dir / "engines" / ".iteration"
    if not it.exists():
        return None
    rows = list(csv.DictReader(it.open(), skipinitialspace=True))
    if not rows:
        return None
    return int(rows[-1]["iteration"]), rows[-1]["engine_path"].strip()


def rate_engine(engine: str, estimate: float | None, ratings: dict, args) -> dict:
    t0 = time.time()
    row: dict = {"puzzles": ladder.puzzle_accuracy(engine, ladder.load_puzzles(), args.sims)}
    openings = ladder.make_openings(args.games // 2)

    def play(name: str) -> list[float]:
        anchor = ladder.Anchor(name, seed=7)
        try:
            return ladder.play_match(engine, anchor, openings, args.sims)
        finally:
            anchor.close()

    picked = ladder.pick_anchors(estimate, ratings, k=args.anchors)
    scores = ladder.adaptive_anchor_scores(play, picked, ratings, sweep=args.sweep, max_extra=args.max_extra)
    per_anchor = {n: {"w": s.count(1.0), "d": s.count(0.5), "l": s.count(0.0)} for n, s in scores.items()}
    results = [(ratings.get(n, 0.0), x) for n, s in scores.items() for x in s]
    elo, se = ladder.fit_rating(results, prior=estimate if estimate is not None else 0.0)
    row.update(elo=round(elo, 1), elo_se=round(se, 1), anchors=per_anchor, extra_anchors=len(scores) - len(picked),
               seconds=round(time.time() - t0, 1))
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--every-min", type=float, default=30.0)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--games", type=int, default=48, help="games per anchor (both colours)")
    p.add_argument("--anchors", type=int, default=2)
    p.add_argument("--sweep", type=float, default=0.85,
                   help="score that counts as a sweep and adds the next anchor up (or down)")
    p.add_argument("--max-extra", type=int, default=3, help="extra anchors per rating (0 = original ladder)")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    if args.calibrate:
        print(json.dumps(ladder.calibrate(), indent=2))
        return

    ratings = ladder.load_anchor_ratings()
    out_dir = args.run_dir / "ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ladder.jsonl"
    estimate, last_t, last_step = None, 0.0, -1
    if out.exists():
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        if rows:
            estimate, last_t, last_step = rows[-1]["elo"], rows[-1]["t"], rows[-1]["step"]
    while True:
        eng = latest_engine(args.run_dir)
        due = time.time() - last_t >= args.every_min * 60
        if eng and due and eng[0] != last_step:
            step, engine = eng
            row = {"t": time.time(), "step": step, "engine": engine, **rate_engine(engine, estimate, ratings, args)}
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"step {step}: elo {row['elo']} ± {row['elo_se']}  puzzles {row['puzzles']['line']:.3f} "
                  f"({row['seconds']}s) {row['anchors']}", flush=True)
            estimate, last_t, last_step = row["elo"], row["t"], step
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
