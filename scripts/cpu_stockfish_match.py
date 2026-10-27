"""The first post's Stockfish test with the network searching on the CPU (src/analysis/engine.py),
for machines without the TensorRT engine. One process per game, several games in parallel.

Stockfish 16 with UCI_LimitStrength + UCI_Elo, 100 ms per move, 1 thread, 16 MB hash (as in
scripts/stockfish_match.py). The network searches 800 simulations per move, one at a time
(cpuct 0.65, most-visited move, no noise), like the GPU test's virtual-loss width 1. Games start
from the initial position with alternating colours and end by the ladder's rules (rules_end).
One JSON row per Stockfish level; all games go to one PGN.

  uv run python scripts/cpu_stockfish_match.py --ckpt step_00100500.pt --elo 2000 2400 --games 40 \
      --workers 8 --out match.jsonl --pgn match.pgn
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.engine import Net, search  # noqa: E402
from src.rating.ladder import rules_end  # noqa: E402

NET: Net | None = None
SF: chess.engine.SimpleEngine | None = None


def init(ckpt: str, stockfish: str) -> None:
    global NET, SF
    NET = Net(ckpt, threads=1)
    SF = chess.engine.SimpleEngine.popen_uci(stockfish)


def play(task: tuple[int, int, bool, int, int, int]) -> dict:
    idx, elo, model_white, sims, movetime_ms, max_plies = task
    SF.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    board = chess.Board()
    t0 = time.time()
    while not rules_end(board)[0] and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == model_white:
            root = search(NET, board, sims, cpuct=0.65, batch=1)["root"]
            move = max(root.children.items(), key=lambda kv: (kv[1].n, kv[1].prior))[0]
        else:
            move = SF.play(board, chess.engine.Limit(time=movetime_ms / 1000.0)).move
        board.push(move)
    winner = rules_end(board)[1]
    score = 0.5 if winner is None else (1.0 if winner == model_white else 0.0)
    game = chess.pgn.Game.from_board(board)
    me, sf = f"ChessRL ({sims} simulations, CPU)", f"Stockfish 16 UCI_Elo {elo} ({movetime_ms} ms)"
    game.headers["Event"] = "CPU Stockfish match"
    game.headers["Round"] = str(idx + 1)
    game.headers["White"], game.headers["Black"] = (me, sf) if model_white else (sf, me)
    game.headers["Result"] = "1/2-1/2" if winner is None else ("1-0" if winner == chess.WHITE else "0-1")
    return {"elo": elo, "score": score, "plies": board.ply(), "seconds": time.time() - t0, "pgn": str(game)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--elo", type=int, nargs="+", required=True)
    p.add_argument("--games", type=int, default=40, help="per Stockfish level, half with each colour")
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--movetime-ms", type=int, default=100)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stockfish", default=shutil.which("stockfish") or "/usr/games/stockfish")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pgn", type=Path, required=True)
    a = p.parse_args()
    tasks = [(i, elo, i % 2 == 0, a.sims, a.movetime_ms, a.max_plies)
             for elo in a.elo for i in range(a.games)]
    per: dict[int, list[float]] = {elo: [] for elo in a.elo}
    with mp.get_context("spawn").Pool(a.workers, initializer=init, initargs=(a.ckpt, a.stockfish)) as pool, \
            a.pgn.open("a") as pgn:
        for r in pool.imap_unordered(play, tasks):
            per[r["elo"]].append(r["score"])
            pgn.write(r["pgn"] + "\n\n")
            pgn.flush()
            s = per[r["elo"]]
            print(f"UCI_Elo {r['elo']}: game {len(s)}/{a.games} {r['score']} ({r['plies']} plies, "
                  f"{r['seconds']:.0f} s), running score {sum(s)}/{len(s)}", flush=True)
    for elo, s in per.items():
        w, d, l = s.count(1.0), s.count(0.5), s.count(0.0)
        frac = sum(s) / len(s)
        implied = elo - 400 * math.log10(1 / frac - 1) if 0 < frac < 1 else None
        row = {"ckpt": a.ckpt, "sf_elo": elo, "sims": a.sims, "movetime_ms": a.movetime_ms, "games": len(s),
               "w": w, "d": d, "l": l, "score": round(frac, 4),
               "implied_elo": round(implied, 1) if implied is not None else None, "where": "cpu"}
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
