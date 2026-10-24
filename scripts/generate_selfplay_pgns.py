"""Generate N self-play games with the live MCTS settings and save a PGN.

Mirrors the self-play loop's MCTS configuration (sims, cpuct, in-tree
temperature, root temperature, Dirichlet noise, random-opening plies)
but plays the games entirely in Python and never touches the on-disk
buffer — so this is safe to run while a training run is using the buffer.

Defaults match src/train/config.yaml. Override on the CLI if needed.

Usage:
    uv run python scripts/generate_selfplay_pgns.py --num-games 4 \\
        --engine artifacts/runs/<id>/engines/step_NNNNNNN.trt \\
        --out artifacts/pgns/selfplay_step_NNNNNNN.pgn
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import random
import sys
import time
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import _inference_cpp  # noqa: E402

from src.rating.ladder import rules_end  # noqa: E402  (automatic threefold, as in self-play)


def latest_engine_from_iteration_file(p: Path) -> Path | None:
    if not p.is_file():
        return None
    last = p.read_text().strip().splitlines()[-1]
    return Path(last.split(",", 1)[1].strip())


def make_pgn(
    history: list[str], result: str, headers: dict, comments: list[str] | None = None
) -> str:
    game = chess.pgn.Game()
    for k, v in headers.items():
        game.headers[k] = v
    game.headers["Result"] = result
    node = game
    for i, u in enumerate(history):
        node = node.add_main_variation(chess.Move.from_uci(u))
        if comments and i < len(comments) and comments[i]:
            node.comment = comments[i]
    return str(game)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine",
        type=Path,
        default=None,
        help="Path to .trt engine. Default: read from artifacts/runs/<latest>/engines/.iteration",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run dir whose engines/.iteration to read. Ignored if --engine is set.",
    )
    ap.add_argument("--num-games", type=int, default=4)
    ap.add_argument("--simulations", type=int, default=800)
    ap.add_argument("--cpuct", type=float, default=0.65)
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.3,
        help="In-tree action-sampling temperature (config.yaml: selfplay.temperature)",
    )
    ap.add_argument(
        "--temperature-at-root",
        type=float,
        default=1.0,
        help="Root move-selection temperature (config.yaml: selfplay.temperature_at_root)",
    )
    ap.add_argument("--use-dirichlet", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.12)
    ap.add_argument("--dirichlet-epsilon", type=float, default=0.05)
    ap.add_argument(
        "--virtual-loss-width",
        type=int,
        default=8,
        help="Match live self-play search (config.yaml: "
        "selfplay.virtual_loss_width). 1 = strict sequential.",
    )
    ap.add_argument(
        "--random-opening-plies",
        type=int,
        default=8,
        help="Uniform-random legal plies before MCTS takes over.",
    )
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PGN. Default: artifacts/pgns/selfplay_<engine>_<ts>.pgn",
    )
    args = ap.parse_args()

    if args.engine is None:
        run_dir = args.run_dir
        if run_dir is None:
            cands = sorted((ROOT / "artifacts/runs").glob("*/engines/.iteration"))
            if not cands:
                raise SystemExit("no engines/.iteration found and --engine not provided")
            iter_path = cands[-1]
        else:
            iter_path = run_dir / "engines" / ".iteration"
        eng = latest_engine_from_iteration_file(iter_path)
        if eng is None:
            raise SystemExit(f"could not read engine path from {iter_path}")
        args.engine = eng

    if not args.engine.is_file():
        raise SystemExit(f"engine not found: {args.engine}")

    if args.out is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = ROOT / "artifacts/pgns" / f"selfplay_{args.engine.stem}_{ts}.pgn"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"engine        : {args.engine}")
    print(f"games         : {args.num_games}")
    print(
        f"MCTS          : sims={args.simulations}  cpuct={args.cpuct}  "
        f"temp={args.temperature}  root_temp={args.temperature_at_root}"
    )
    print(
        f"Dirichlet     : enabled={args.use_dirichlet}  alpha={args.dirichlet_alpha}  "
        f"eps={args.dirichlet_epsilon}"
    )
    print(f"random opening: {args.random_opening_plies} plies")
    print(f"output PGN    : {args.out}")
    print()

    rng = random.Random(args.seed)

    # ---- per-game state ----
    boards = [chess.Board() for _ in range(args.num_games)]
    histories: list[list[str]] = [[] for _ in range(args.num_games)]
    # Per-move PGN comment. Opening (random) plies get no eval (self-play
    # doesn't search them); MCTS plies get the root WDL from the moving
    # side's POV — i.e. what the search thought of the position it moved in.
    comments: list[list[str]] = [[] for _ in range(args.num_games)]
    finished = [False] * args.num_games
    results: list[str] = [""] * args.num_games

    # ---- random-opening plies (uniform legal sampling) ----
    for g in range(args.num_games):
        for _ in range(args.random_opening_plies):
            if rules_end(boards[g])[0]:
                break
            mv = rng.choice(list(boards[g].legal_moves))
            histories[g].append(mv.uci())
            comments[g].append("")  # random opening, no MCTS eval
            boards[g].push(mv)

    # ---- main loop: batch MCTS across the live games ----
    t0 = time.time()
    ply = max(args.random_opening_plies, 0)
    while not all(finished):
        active = [g for g in range(args.num_games) if not finished[g]]
        if not active:
            break

        batch_histories = [histories[g] for g in active]
        batch_fens = ["" for _ in active]

        res = _inference_cpp.small_mcts_search_batch_with_history(
            histories=batch_histories,
            fens=batch_fens,
            simulations=args.simulations,
            cpuct=args.cpuct,
            temperature=args.temperature,
            engine_path=str(args.engine),
            temperature_at_root=args.temperature_at_root,
            use_dirichlet=args.use_dirichlet,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
            virtual_loss_width=args.virtual_loss_width,
        )

        for k, g in enumerate(active):
            uci = res[k]["action"]
            mv = chess.Move.from_uci(uci)
            if mv not in boards[g].legal_moves:
                # Defensive: treat as resign by side-to-move.
                results[g] = "0-1" if boards[g].turn == chess.WHITE else "1-0"
                finished[g] = True
                print(f"  game {g + 1}: illegal move {uci} at ply {len(histories[g])}; resigning")
                continue
            histories[g].append(uci)
            val = res[k].get("value")  # [W, D, L], moving side's POV
            if val and len(val) == 3:
                w, d, l = float(val[0]), float(val[1]), float(val[2])
                comments[g].append(
                    f"mcts W={w:.3f} D={d:.3f} L={l:.3f} v={w - l:+.3f} ply{len(histories[g])}"
                )
            else:
                comments[g].append("")
            boards[g].push(mv)

            over, winner = rules_end(boards[g])
            if over or len(histories[g]) >= args.max_plies:
                if winner is None:
                    results[g] = "1/2-1/2"
                else:
                    results[g] = "1-0" if winner == chess.WHITE else "0-1"
                finished[g] = True
                print(
                    f"  game {g + 1} done  result={results[g]}  plies={len(histories[g])}  "
                    f"elapsed={time.time() - t0:.0f}s"
                )
        ply += 1

    elapsed = time.time() - t0
    print(
        f"\nall {args.num_games} games done in {elapsed:.0f}s "
        f"(~{elapsed / args.num_games:.1f}s/game wall-clock with batching)"
    )

    # ---- write PGN ----
    pgn = io.StringIO()
    date = dt.date.today().strftime("%Y.%m.%d")
    for g in range(args.num_games):
        headers = {
            "Event": "Self-play",
            "Site": "local",
            "Date": date,
            "Round": str(g + 1),
            "White": args.engine.stem,
            "Black": args.engine.stem,
        }
        pgn.write(make_pgn(histories[g], results[g], headers, comments[g]) + "\n\n")
    args.out.write_text(pgn.getvalue())
    print(f"PGN: {args.out}")
    # Quick summary
    counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for r in results:
        counts[r] = counts.get(r, 0) + 1
    print(
        f"results: 1-0={counts.get('1-0', 0)}  0-1={counts.get('0-1', 0)}  "
        f"draws={counts.get('1/2-1/2', 0)}"
    )


if __name__ == "__main__":
    main()
