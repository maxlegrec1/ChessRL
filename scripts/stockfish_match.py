"""Play N games of MCTS-N model vs Stockfish at fixed Elo.

Saves a single PGN with all games + a summary line.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import _inference_cpp


def get_model_move(
    history: list[str],
    engine_path: str,
    simulations: int,
    cpuct: float,
    virtual_loss_width: int = 1,
) -> str:
    res = _inference_cpp.small_mcts_search_batch_with_history(
        histories=[history],
        fens=[""],
        simulations=simulations,
        cpuct=cpuct,
        temperature=1.0,
        engine_path=engine_path,
        temperature_at_root=0.0,
        use_dirichlet=False,
        virtual_loss_width=virtual_loss_width,
    )
    return res[0]["action"]


def play_one(
    sf,
    engine_path,
    model_white,
    simulations,
    cpuct,
    sf_movetime_ms,
    max_plies,
    game_idx,
    virtual_loss_width=1,
):
    board = chess.Board()
    history: list[str] = []
    while not board.is_game_over(claim_draw=True) and len(history) < max_plies:
        is_model_turn = (board.turn == chess.WHITE) == model_white
        if is_model_turn:
            uci = get_model_move(history, engine_path, simulations, cpuct, virtual_loss_width)
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                print(f"  [game {game_idx}] illegal model move {uci}; resign", flush=True)
                return board, history, ("0-1" if model_white else "1-0")
        else:
            result = sf.play(board, chess.engine.Limit(time=sf_movetime_ms / 1000.0))
            mv = result.move
        history.append(mv.uci())
        board.push(mv)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return board, history, "1/2-1/2"
    return board, history, ("1-0" if outcome.winner == chess.WHITE else "0-1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to TRT engine .trt")
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--num-games", type=int, default=20)
    ap.add_argument("--simulations", type=int, default=800)
    ap.add_argument("--cpuct", type=float, default=0.65)
    ap.add_argument("--sf-elo", type=int, default=1320)
    ap.add_argument("--sf-movetime-ms", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument(
        "--virtual-loss-width",
        type=int,
        default=1,
        help="MCTS virtual-loss batched-descent width. Use 1 to disable.",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.out is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = Path(args.engine).stem
        args.out = ROOT / "artifacts/pgns" / f"sf_{args.sf_elo}_vs_{ckpt_name}_{ts}.pgn"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"engine:      {args.engine}")
    print(f"stockfish:   {args.stockfish}  (Elo={args.sf_elo}, movetime={args.sf_movetime_ms}ms)")
    print(f"games:       {args.num_games}  (MCTS-{args.simulations}, cpuct={args.cpuct})")
    print(f"output PGN:  {args.out}")
    print()

    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo})

    games_pgn = io.StringIO()
    score = 0.0
    wins = draws = losses = 0
    start = time.time()

    try:
        for i in range(args.num_games):
            model_white = i % 2 == 0
            t0 = time.time()
            board, history, result = play_one(
                sf,
                args.engine,
                model_white,
                args.simulations,
                args.cpuct,
                args.sf_movetime_ms,
                args.max_plies,
                i + 1,
                virtual_loss_width=args.virtual_loss_width,
            )
            dt_ = time.time() - t0

            if result == "1-0":
                model_pts = 1.0 if model_white else 0.0
            elif result == "0-1":
                model_pts = 0.0 if model_white else 1.0
            else:
                model_pts = 0.5
            score += model_pts
            if model_pts == 1.0:
                wins += 1
            elif model_pts == 0.5:
                draws += 1
            else:
                losses += 1

            game = chess.pgn.Game()
            game.headers["Event"] = "Model vs Stockfish"
            game.headers["Site"] = "local"
            game.headers["Date"] = dt.date.today().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i + 1)
            game.headers["White"] = (
                Path(args.engine).stem if model_white else f"Stockfish-{args.sf_elo}"
            )
            game.headers["Black"] = (
                f"Stockfish-{args.sf_elo}" if model_white else Path(args.engine).stem
            )
            game.headers["Result"] = result
            node = game
            tmp = chess.Board()
            for u in history:
                mv = chess.Move.from_uci(u)
                node = node.add_main_variation(mv)
                tmp.push(mv)
            games_pgn.write(str(game) + "\n\n")

            print(
                f"  [{i + 1:>2}/{args.num_games}] {result}  "
                f"model={'W' if model_white else 'B'}  pts={model_pts}  "
                f"plies={len(history)}  time={dt_:.1f}s  "
                f"running: {wins}W/{draws}D/{losses}L  ({100 * score / (i + 1):.1f}%)",
                flush=True,
            )
    finally:
        sf.quit()

    elapsed = time.time() - start
    n = args.num_games
    win_rate = score / n
    args.out.write_text(games_pgn.getvalue())

    print()
    print("=== Final ===")
    print(
        f"Model vs Stockfish-{args.sf_elo}: {wins}W / {draws}D / {losses}L  ->  {score}/{n}  ({100 * win_rate:.1f}%)"
    )
    if 0.0 < win_rate < 1.0:
        elo_diff = -400 * math.log10(1 / win_rate - 1)
        print(
            f"Estimated model Elo ~ {args.sf_elo} + ({elo_diff:+.0f})  =  {args.sf_elo + elo_diff:.0f}"
        )
    print(f"Total time: {elapsed:.0f}s")
    print(f"PGN: {args.out}")


if __name__ == "__main__":
    main()
