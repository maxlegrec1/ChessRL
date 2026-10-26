"""Showcase games: a network against one ladder opponent, saved as PGN.

The network searches as in the first post's Stockfish test (800 simulations per move, cpuct 0.65,
most-visited move, no noise). The opponent is a ladder anchor, by default full-strength Stockfish
limited to 1,000 nodes per move. Games start from the initial position and from the first N
balanced Stockfish openings, each with both colours. Both sides are deterministic, so the
balanced openings are what makes the games differ.

  uv run python scripts/showcase_games.py --engine c800.trt --anchor sf_s20_n1000 --openings 4 --out games.pgn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rating import ladder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, help="TensorRT engine of the network")
    p.add_argument("--name", default=None, help="the network's name in the PGN (default: engine file stem)")
    p.add_argument("--anchor", default="sf_s20_n1000", choices=sorted(ladder.ANCHORS))
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--openings", type=int, default=4, help="balanced openings, besides the initial position")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    name = a.name or Path(a.engine).stem
    openings = [[]] + ladder.load_openings("sf", a.openings)
    games = [ladder.Game(o, w) for o in openings for w in (True, False)]
    anchor = ladder.Anchor(a.anchor, seed=7)
    try:
        # Same loop as ladder.play_match, keeping the boards.
        while True:
            for g in games:
                if g.result is None:
                    over, winner = ladder.rules_end(g.board)
                    if over or len(g.history) >= a.max_plies:
                        g.result = 0.5 if winner is None else (1.0 if winner == g.model_white else 0.0)
            live = [g for g in games if g.result is None]
            if not live:
                break
            to_search = [g for g in live if g.model_to_move]
            if to_search:
                for g, uci in zip(to_search, ladder.mcts_moves(a.engine, to_search, a.sims), strict=True):
                    g.board.push_uci(uci)
                    g.history.append(uci)
            for g in live:
                if not g.model_to_move and not ladder.rules_end(g.board)[0]:
                    m = anchor.move(g.board)
                    g.board.push(m)
                    g.history.append(m.uci())
    finally:
        anchor.close()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for i, g in enumerate(games):
            pgn = chess.pgn.Game.from_board(g.board)
            me, opp = f"{name} ({a.sims} simulations)", f"{a.anchor} (ladder anchor)"
            pgn.headers["Event"] = "ChessRL showcase"
            pgn.headers["Round"] = str(i + 1)
            pgn.headers["White"], pgn.headers["Black"] = (me, opp) if g.model_white else (opp, me)
            pgn.headers["Opening"] = "initial position" if not g.opening else f"balanced: {' '.join(g.opening)}"
            if g.result == 0.5:
                pgn.headers["Result"] = "1/2-1/2"
            else:
                white_won = (g.result == 1.0) == g.model_white
                pgn.headers["Result"] = "1-0" if white_won else "0-1"
            print(pgn, file=fh, end="\n\n")
            print(f"game {i + 1}: model {'white' if g.model_white else 'black'}, {pgn.headers['Opening'][:40]}, "
                  f"{pgn.headers['Result']} in {g.board.fullmove_number} moves", flush=True)
    score = sum(g.result for g in games)
    print(f"model score {score}/{len(games)} vs {a.anchor}", flush=True)


if __name__ == "__main__":
    main()
