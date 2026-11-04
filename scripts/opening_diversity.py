"""Opening diversity of self-play: how varied the positions are at fixed plies of the game, from the newest records
of a run's buffer. Meant to catch a collapse onto a few openings once openings are sampled instead of random.

Per absolute ply (records are canonical, so the ply comes from the fullmove number and FLAG_BLACK_TO_MOVE), over the
records at that ply in the window:
  - effective positions: 2 ** entropy of the position counts (equals the number of distinct positions when they
    are equally frequent; a collapse onto a few openings drives it down);
  - top share: the share of the most common position;
  - singletons: the share of records whose position appears once in the window.
A position is the FEN's board, side, castling and en passant fields. Records exist only for full searches (25% of
moves), so each game contributes about a quarter of its positions.

    uv run python scripts/opening_diversity.py RUN_DIR [--window 400000] [--plies 8 12 16 20 30] [--out FILE]
    uv run python scripts/opening_diversity.py RUN_DIR --every-min 10   # loop, appending to metrics/diversity.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.dataset import FLAG_BLACK_TO_MOVE, RECORD_DTYPE, RECORD_VERSION  # noqa: E402


def newest(buffer: Path, window: int) -> np.ndarray:
    mm = np.memmap(buffer, dtype=RECORD_DTYPE, mode="r")
    cap = len(mm)
    writes_path = buffer.with_name(buffer.name + ".writes")
    writes = int(writes_path.read_text().split()[0]) if writes_path.exists() else cap
    end = writes % cap if writes > cap else min(writes, cap)
    n = min(window, cap, writes)
    rows = np.asarray(mm[(end - 1 - np.arange(n)) % cap])
    return rows[rows["version"] == RECORD_VERSION]


def diversity(rows: np.ndarray, plies: list[int]) -> dict:
    black = (rows["flags"] & FLAG_BLACK_TO_MOVE) > 0
    ply = 2 * (rows["fullmove"].astype(np.int64) - 1) + black
    out = {}
    for p in plies:
        sel = rows[ply == p]
        if len(sel) < 20:
            continue
        keys = Counter(b" ".join(f.split(b" ")[:4]) for f in sel["fen"])
        n = len(sel)
        probs = np.array(list(keys.values()), dtype=np.float64) / n
        entropy = float(-(probs * np.log2(probs)).sum())
        out[str(p)] = {
            "records": n,
            "effective": round(2.0**entropy, 1),
            "top_share": round(float(probs.max()), 4),
            "singletons": round(sum(c for c in keys.values() if c == 1) / n, 4),
            "norm_entropy": round(entropy / math.log2(n), 4) if n > 1 else 1.0,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path, help="run directory (or a buffer file)")
    p.add_argument("--window", type=int, default=400_000, help="newest records to look at")
    p.add_argument("--plies", type=int, nargs="+", default=[8, 12, 16, 20, 30])
    p.add_argument(
        "--out", type=Path, default=None, help="append one JSON line here (default: print)"
    )
    p.add_argument(
        "--every-min",
        type=float,
        default=0.0,
        help="repeat every N minutes into metrics/diversity.jsonl",
    )
    a = p.parse_args()
    buffer = a.run if a.run.is_file() else a.run / "buffer" / "buffer.bin"
    out = a.out or (a.run / "metrics" / "diversity.jsonl" if a.every_min > 0 else None)
    while True:
        if buffer.exists():
            row = {
                "t": time.time(),
                "window": a.window,
                "plies": diversity(newest(buffer, a.window), a.plies),
            }
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
            else:
                print(json.dumps(row, indent=1))
        if a.every_min <= 0:
            return
        time.sleep(a.every_min * 60)


if __name__ == "__main__":
    main()
