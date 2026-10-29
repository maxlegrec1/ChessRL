"""Write data/bench/positions.txt, the positions of the CPU search benchmark.

Stockfish plays itself from the first --games balanced openings (data/openings), one
thread at a fixed node count, so the games (and the file) are reproducible. Each game
contributes its positions at the plies in PLIES, written as the game's UCI moves from the
standard start: the benchmark replays them, so every position carries its real history
(a search copies the game's past states, so late positions cost more).

    uv run python scripts/make_bench_positions.py
"""

import argparse
import json
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[1]
PLIES = [8, 14, 20, 28, 36, 46, 58, 72, 90, 115, 140]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--nodes", type=int, default=30_000)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--out", type=Path, default=ROOT / "data/bench/positions.txt")
    a = ap.parse_args()

    openings = json.loads((ROOT / "data/openings/sf_balanced_6ply.json").read_text())["openings"]
    engine = chess.engine.SimpleEngine.popen_uci(a.stockfish)
    engine.configure({"Threads": 1, "Hash": 16})
    version = engine.id.get("name", "stockfish")
    lines = []
    try:
        for g in range(a.games):
            board = chess.Board()
            for uci in openings[g]["moves"]:
                board.push_uci(uci)
            while not board.is_game_over() and board.ply() < a.max_plies:
                board.push(engine.play(board, chess.engine.Limit(nodes=a.nodes)).move)
            moves = [m.uci() for m in board.move_stack]
            for ply in PLIES:
                if ply < len(moves):
                    lines.append(" ".join(moves[:ply]))
            print(f"game {g + 1}/{a.games}: {len(moves)} plies, {board.result()}", flush=True)
    finally:
        engine.quit()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# CPU search benchmark positions (scripts/make_bench_positions.py).",
        f"# {version} self-play, {a.nodes} nodes per move, from the first {a.games} openings of",
        f"# data/openings/sf_balanced_6ply.json; positions at plies {PLIES}.",
        "# One position per line: its game's UCI moves from the standard start.",
    ]
    a.out.write_text("\n".join(header + lines) + "\n")
    print(f"wrote {len(lines)} positions to {a.out}")


if __name__ == "__main__":
    main()
