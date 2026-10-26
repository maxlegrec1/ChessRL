"""Balanced openings for the rating ladder, chosen by Stockfish (no opening book, no random moves).

From the initial position, each ply is chosen among the moves Stockfish rates within --margin
centipawns of its best (MultiPV analysis, fixed node budget), with probability proportional to
exp((score - best) / --softness): the best move most often, near-equal alternatives regularly.
(Uniform choice at 30 cp produced balanced but odd lines such as 1.e3 d6 2.h4 h6 3.a3 a6: from
the initial position Stockfish rates most quiet moves within 30 cp.) An opening is kept if the
final position is within --balance centipawns and the move sequence is new. The set is generated
once and stored, so every box rates on the same openings.

  uv run python scripts/make_sf_openings.py --out data/openings/sf_balanced_6ply.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rating.ladder import stockfish_path  # noqa: E402


def _worker(args: tuple[int, int, int, int, int, int, int, float]) -> list[dict]:
    seed, count, plies, margin, balance, nodes, multipv, softness = args
    rng = random.Random(seed)
    eng = chess.engine.SimpleEngine.popen_uci(stockfish_path())
    eng.configure({"Threads": 1, "Hash": 64})
    out, tries = [], 0
    try:
        while len(out) < count and tries < count * 6:
            tries += 1
            board = chess.Board()
            for _ in range(plies):
                infos = eng.analyse(board, chess.engine.Limit(nodes=nodes), multipv=multipv)
                scored = [(i["pv"][0], i["score"].relative.score(mate_score=10000)) for i in infos if "pv" in i]
                best = max(s for _, s in scored)
                cands = [(m, math.exp((s - best) / softness)) for m, s in scored if s >= best - margin]
                board.push(rng.choices([m for m, _ in cands], weights=[w for _, w in cands])[0])
            final = eng.analyse(board, chess.engine.Limit(nodes=nodes))["score"].white().score(mate_score=10000)
            if abs(final) <= balance:
                moves = [m.uci() for m in board.move_stack]
                out.append({"moves": moves, "san": chess.Board().variation_san(board.move_stack), "eval_cp": final})
    finally:
        eng.quit()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--plies", type=int, default=6)
    p.add_argument("--margin", type=int, default=20, help="centipawns below the best move still eligible")
    p.add_argument("--softness", type=float, default=10.0, help="centipawns per e-fold of choice weight")
    p.add_argument("--balance", type=int, default=40, help="max |eval| of the final position, centipawns")
    p.add_argument("--nodes", type=int, default=1_000_000)
    p.add_argument("--multipv", type=int, default=8)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    per = a.n // a.procs + 4  # a few spare, duplicates are dropped below
    jobs = [(a.seed + i, per, a.plies, a.margin, a.balance, a.nodes, a.multipv, a.softness) for i in range(a.procs)]
    with mp.Pool(a.procs) as pool:
        found = [o for part in pool.map(_worker, jobs) for o in part]
    seen, uniq = set(), []
    for o in found:
        key = tuple(o["moves"])
        if key not in seen:
            seen.add(key)
            uniq.append(o)
    random.Random(a.seed).shuffle(uniq)
    uniq = uniq[: a.n]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: getattr(a, k) for k in ("plies", "margin", "softness", "balance", "nodes", "multipv", "seed")}
    a.out.write_text(json.dumps({"generator": meta, "openings": uniq}, indent=1) + "\n")
    first = {}
    for o in uniq:
        first[o["moves"][0]] = first.get(o["moves"][0], 0) + 1
    print(f"wrote {len(uniq)} openings to {a.out}; first moves {dict(sorted(first.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
