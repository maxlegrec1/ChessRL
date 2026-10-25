"""Opening diversity of self-play games (PGN files written with selfplay.save_pgn).

For the earliest and the latest N games of a run: the share of the most common
first moves, and how many distinct positions there are after 4, 8 and 12 plies.

  uv run python scripts/opening_diversity.py <run_dir>/pgns [--n 3000]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import chess
import chess.pgn


def openings(files: list[Path], plies: int = 12) -> list[list[str]]:
    out = []
    for f in files:
        with f.open() as fh:
            game = chess.pgn.read_game(fh)
        if game is None:
            continue
        moves = []
        for i, mv in enumerate(game.mainline_moves()):
            if i >= plies:
                break
            moves.append(mv.uci())
        out.append(moves)
    return out


def stats(lines: list[list[str]]) -> dict:
    n = len(lines)
    first = collections.Counter(m[0] for m in lines if m)
    res = {"games": n, "top_first_moves": {k: round(v / n, 3) for k, v in first.most_common(5)},
           "distinct_first_moves": len(first)}
    for p in (4, 8, 12):
        seqs = collections.Counter(tuple(m[:p]) for m in lines if len(m) >= p)
        board_keys = set()
        for m in lines:
            if len(m) < p:
                continue
            b = chess.Board()
            for u in m[:p]:
                b.push_uci(u)
            board_keys.add(b.board_fen() + (" w" if b.turn else " b"))
        res[f"ply{p}"] = {"distinct_positions": len(board_keys), "distinct_share": round(len(board_keys) / n, 3),
                          "top_line_share": round(seqs.most_common(1)[0][1] / n, 3) if seqs else 0.0}
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pgn_dir", type=Path)
    p.add_argument("--n", type=int, default=3000)
    a = p.parse_args()
    files = sorted(a.pgn_dir.glob("game_*.pgn"), key=lambda f: int(f.stem.split("_")[1]))
    out = {"total_games": len(files), "earliest": stats(openings(files[: a.n])),
           "latest": stats(openings(files[-a.n:]))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
