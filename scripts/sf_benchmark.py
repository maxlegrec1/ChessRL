"""The first post's Stockfish benchmark, for several checkpoints and Stockfish levels in parallel.

Protocol (unchanged from scripts/sf1400_autobrawl.py, which called stockfish_match.py with its
defaults): Stockfish 16 with UCI_LimitStrength + UCI_Elo, 100 ms per move; the network searches
800 simulations per move (cpuct 0.65, virtual-loss width 1, most-visited move); games from the
initial position, alternating colours. The implied Elo is the Stockfish level plus the Elo
difference for the score, as in the first post.

  uv run python scripts/sf_benchmark.py --engines a.trt b.trt --elo 1400 2000 --games 100 --out bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_match(engine: str, elo: int, games: int, sims: int, pgn_dir: Path) -> dict:
    pgn = pgn_dir / f"sf{elo}_{Path(engine).stem}.pgn"
    cmd = [sys.executable, "scripts/stockfish_match.py", "--engine", engine, "--num-games", str(games),
           "--sf-elo", str(elo), "--simulations", str(sims), "--out", str(pgn)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+)W / (\d+)D / (\d+)L\s+->\s+([\d.]+)/(\d+)", out)
    if not m:
        return {"engine": engine, "sf_elo": elo, "error": out[-500:]}
    w, d, l, score, n = int(m[1]), int(m[2]), int(m[3]), float(m[4]), int(m[5])
    p = score / n
    implied = elo - 400 * math.log10(1 / p - 1) if 0 < p < 1 else None
    return {"engine": Path(engine).stem, "sf_elo": elo, "sims": sims, "games": n, "w": w, "d": d, "l": l,
            "score": round(p, 4), "implied_elo": round(implied, 1) if implied is not None else None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engines", nargs="+", required=True)
    p.add_argument("--elo", type=int, nargs="+", default=[1400])
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pgn-dir", type=Path, default=Path("/workspace/sf_benchmark_pgns"))
    a = p.parse_args()
    a.pgn_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(e, elo) for elo in a.elo for e in a.engines]
    with ThreadPoolExecutor(a.parallel) as pool:
        for row in pool.map(lambda j: run_match(j[0], j[1], a.games, a.sims, a.pgn_dir), jobs):
            with a.out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
