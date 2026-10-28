"""Quality of one side's play in saved games: Stockfish's evaluation loss per move.

For every game of a PGN won (or drawn, with --draws) by the player whose name contains --player,
evaluates each of that player's moves with Stockfish at a fixed depth and reports the average
loss in centipawns, the largest loss and the number of moves that lose 100+ centipawns. Used to
pick a decent game to show (docs/blog/make_boards.py).

  uv run python scripts/game_quality.py --pgn games.pgn --player ChessRL --depth 14
"""

from __future__ import annotations

import argparse
import shutil

import chess
import chess.engine
import chess.pgn

MATE = 10_000


def cp(info: dict, color: chess.Color) -> int:
    return info["score"].pov(color).score(mate_score=MATE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", required=True)
    p.add_argument("--player", default="ChessRL")
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--draws", action="store_true")
    p.add_argument("--filter", default="", help="only games whose opponent name contains this")
    a = p.parse_args()
    sf = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/usr/games/stockfish")
    sf.configure({"Threads": 2, "Hash": 64})
    limit = chess.engine.Limit(depth=a.depth)
    rows = []
    with open(a.pgn) as fh:
        idx = 0
        while (g := chess.pgn.read_game(fh)) is not None:
            idx += 1
            white = a.player in g.headers["White"]
            opp = g.headers["Black" if white else "White"]
            if a.filter not in opp:
                continue
            res = g.headers["Result"]
            won = res == ("1-0" if white else "0-1")
            if not (won or (a.draws and res == "1/2-1/2")):
                continue
            me = chess.WHITE if white else chess.BLACK
            board = g.board()
            losses = []
            for move in g.mainline_moves():
                if board.turn == me:
                    best = cp(sf.analyse(board, limit), me)
                    board.push(move)
                    after = cp(sf.analyse(board, limit), me)
                    losses.append(max(0, min(best, 1000) - min(after, 1000)))
                else:
                    board.push(move)
            rows.append(
                (
                    idx,
                    opp,
                    res,
                    board.fullmove_number,
                    sum(losses) / max(1, len(losses)),
                    max(losses or [0]),
                    sum(x >= 100 for x in losses),
                )
            )
    sf.quit()
    print(
        "game  opponent                                   result moves  avg_loss  max_loss  >=100cp"
    )
    for r in sorted(rows, key=lambda r: r[4]):
        print(f"{r[0]:4d}  {r[1][:42]:42s} {r[2]:7s} {r[3]:5d}  {r[4]:8.1f}  {r[5]:8d}  {r[6]:7d}")


if __name__ == "__main__":
    main()
