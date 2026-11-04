"""Re-rate saved checkpoints on a given opening set (e.g. after switching the ladder to Stockfish openings).

For each checkpoint: export it (.cnet) with its own search heads, play the same
rating as the ladder daemon (two anchors around a starting estimate, 48 games each, stepping up
or down on lopsided results), and append a row to --out. The estimate is the run's old ladder
rating at that step when available, else --estimate.

  uv run python scripts/rerate.py --openings sf --out rerate_sf.jsonl --ckpt runs/<id>/checkpoints/step_00060000.pt ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export.to_cnet import export  # noqa: E402
from src.rating import ladder  # noqa: E402


def old_rating(ckpt: Path) -> float | None:
    f = ckpt.parent.parent / "ladder" / "ladder.jsonl"
    if not f.exists():
        return None
    step = int(ckpt.stem.split("_")[1])
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    near = [r for r in rows if "elo" in r and abs(r["step"] - step) <= 2500]
    return min(near, key=lambda r: abs(r["step"] - step))["elo"] if near else None


def build(ckpt: Path, work: Path) -> str:
    net = work / f"{ckpt.parent.parent.name}_{ckpt.stem}.cnet"
    if not net.exists():
        export(ckpt, net)
    return str(net)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--openings", choices=("sf", "random"), default="sf")
    p.add_argument("--games", type=int, default=48, help="games per anchor (both colours)")
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--estimate", type=float, default=1500.0)
    p.add_argument("--work", type=Path, default=Path("/workspace/rerate_engines"))
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    ratings = ladder.load_anchor_ratings(a.openings)
    if not ratings:
        sys.exit(f"no anchor calibration for the {a.openings!r} openings")
    openings = ladder.load_openings(a.openings, a.games // 2)
    done = set()
    if a.out.exists():
        done = {
            json.loads(line)["checkpoint"]
            for line in a.out.read_text().splitlines()
            if line.strip()
        }
    for ck in a.ckpt:
        if str(ck) in done:
            continue
        t0 = time.time()
        engine = build(ck, a.work)
        old = old_rating(ck)
        estimate = old if old is not None else a.estimate

        def play(name: str, engine: str = engine) -> list[float]:
            anchor = ladder.Anchor(name, seed=7)
            try:
                return ladder.play_match(engine, anchor, openings, a.sims)
            finally:
                anchor.close()

        picked = ladder.pick_anchors(estimate, ratings, k=2)
        scores = ladder.adaptive_anchor_scores(play, picked, ratings)
        results = [(ratings[n], x) for n, s in scores.items() for x in s]
        elo, se = ladder.fit_rating(results, prior=estimate)
        row = {
            "checkpoint": str(ck),
            "run": ck.parent.parent.name,
            "step": int(ck.stem.split("_")[1]),
            "openings": a.openings,
            "elo": round(elo, 1),
            "elo_se": round(se, 1),
            "old_elo": old,
            "anchors": {
                n: {"w": s.count(1.0), "d": s.count(0.5), "l": s.count(0.0)}
                for n, s in scores.items()
            },
            "seconds": round(time.time() - t0, 1),
        }
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"{row['run']} step {row['step']}: {row['elo']} ± {row['elo_se']} (old {old})",
            flush=True,
        )


if __name__ == "__main__":
    main()
