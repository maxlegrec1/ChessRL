"""MCTS-vs-MCTS brawl between two TRT engines.

Plays a tournament of N games between two checkpoints' TRT engines using
the C++ MCTS harness (`_inference_cpp.small_mcts_search_batch_with_history`).
Each engine plays half the games as White, half as Black so the result is
balanced. Per-game opening positions are sampled from the existing HF
opening generator (same as `brawls.py`).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.eval.mcts_brawl \\
        --engine-a artifacts/runs/.../engines/step_00008200.trt \\
        --engine-b artifacts/runs/.../engines/step_00007700.trt \\
        --num-games 50 --simulations 600 --temp-root 0.2

Notes:
- Uses CUDA_VISIBLE_DEVICES=0 (the 4090). Shares the GPU with any live
  training/self-play; brawls use modest VRAM but compete for compute.
- Auto-detects engines: pass `--run-dir <run>` to brawl latest vs latest-500.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import _inference_cpp
import chess
from chess import pgn


def _load_openings(seed: int):
    try:
        from src.generate.openings import hf_opening_fen_generator as gen
    except Exception:
        return None
    try:
        return gen(split="train", seed=seed)
    except Exception:
        return None


def _opening_to_state(opening) -> tuple[str, list[str]]:
    """Return (fen, history_ucis) for an opening dict from the HF generator.

    Histories track every ply played from STARTPOS so the C++ MCTS sees the
    full move sequence. When the opening provides only a FEN (no UCIs), we
    treat it as an absolute starting position with no history.
    """
    fen = opening.get("fen") if isinstance(opening, dict) else None
    moves = opening.get("uci") if isinstance(opening, dict) else None
    if fen and moves:
        # Validate
        b = chess.Board(fen)
        # If moves are provided, they're played starting from STARTPOS
        # to reach this fen. Replay from start to build history.
        replay = chess.Board()
        history = []
        for u in moves:
            try:
                m = chess.Move.from_uci(u)
                if m in replay.legal_moves:
                    replay.push(m)
                    history.append(u)
                else:
                    break
            except ValueError:
                break
        # If replay matches the provided fen, use it; otherwise fall back.
        if replay.fen().split()[:4] == b.fen().split()[:4]:
            return replay.fen(), history
        # Mismatch: treat fen as standalone
        return fen, []
    if fen:
        return fen, []
    return chess.STARTING_FEN, []


def _result_for_white(board: chess.Board) -> str | None:
    """Return 'W'/'D'/'L' for white's perspective, or None if game continues."""
    if board.is_checkmate():
        # The side to move was just checkmated → that side LOST.
        return "L" if board.turn == chess.WHITE else "W"
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_repetition(3)
    ):
        return "D"
    if board.fullmove_number > 300:  # 600-ply cap as a safety
        return "D"
    return None


def play_brawl(
    engine_a: str,
    engine_b: str,
    num_games: int,
    simulations: int,
    cpuct: float,
    temp: float,
    temp_root: float,
    seed: int,
    pgn_path: Path | None = None,
) -> dict:
    openings = _load_openings(seed)

    # Build games: alternate which engine is white.
    # Game i has engine_a as white when i is even.
    boards: list[chess.Board] = []
    histories: list[list[str]] = []
    engine_for_white: list[str] = []
    finished_results: list[str | None] = []
    for i in range(num_games):
        opening = next(openings) if openings is not None else None
        if opening is None:
            fen, hist = chess.STARTING_FEN, []
        else:
            fen, hist = _opening_to_state(opening)
        try:
            b = chess.Board(fen)
        except Exception:
            b = chess.Board()
            hist = []
        boards.append(b)
        histories.append(hist)
        engine_for_white.append(engine_a if i % 2 == 0 else engine_b)
        finished_results.append(None)

    # Quickly check any games that are already in a terminal state from the opening.
    for i, b in enumerate(boards):
        r = _result_for_white(b)
        if r is not None:
            finished_results[i] = r

    print(f"[brawl] engines: A={Path(engine_a).name}  B={Path(engine_b).name}", flush=True)
    print(
        f"[brawl] {num_games} games | sims={simulations} cpuct={cpuct} temp={temp} temp_root={temp_root}",
        flush=True,
    )
    print("[brawl] starting...", flush=True)

    start_time = time.time()
    ply = 0
    while True:
        active = [i for i, r in enumerate(finished_results) if r is None]
        if not active:
            break
        ply += 1
        if ply > 600:
            for i in active:
                finished_results[i] = "D"
            break

        # Partition active games by which engine to query for the current side-to-move.
        partition: dict = {engine_a: [], engine_b: []}
        for i in active:
            b = boards[i]
            white_engine = engine_for_white[i]
            black_engine = engine_b if white_engine == engine_a else engine_a
            engine_now = white_engine if b.turn == chess.WHITE else black_engine
            partition[engine_now].append(i)

        # For each engine, batch the MCTS call.
        # IMPORTANT: the C++ Python entry point's apply_history replays the
        # history on top of the FEN. To reach the current position, we pass
        # an EMPTY FEN (→ defaults to STARTPOS) and the full move history
        # from startpos. Self-play avoids this by passing a chess::Board
        # object directly via the C++-only path; we don't have that hook
        # from Python.
        for engine_path, indices in partition.items():
            if not indices:
                continue
            sub_histories = [histories[i] for i in indices]
            sub_fens = [""] * len(indices)
            try:
                summaries = _inference_cpp.small_mcts_search_batch_with_history(
                    histories=sub_histories,
                    fens=sub_fens,
                    simulations=simulations,
                    cpuct=cpuct,
                    temperature=temp,
                    engine_path=engine_path,
                    temperature_at_root=temp_root,
                    use_dirichlet=False,
                )
            except Exception as exc:
                print(f"[brawl] MCTS error on {engine_path}: {exc}", file=sys.stderr)
                for i in indices:
                    finished_results[i] = "D"
                continue
            for i, s in zip(indices, summaries, strict=True):
                action = s.get("action", "") if hasattr(s, "get") else dict(s).get("action", "")
                if not action:
                    finished_results[i] = "D"
                    continue
                # Apply move; mark result on game-over.
                try:
                    mv = chess.Move.from_uci(action)
                except ValueError:
                    finished_results[i] = "D"
                    continue
                if mv not in boards[i].legal_moves:
                    finished_results[i] = "D"
                    continue
                boards[i].push(mv)
                histories[i].append(action)
                r = _result_for_white(boards[i])
                if r is not None:
                    finished_results[i] = r

        if ply % 5 == 0:
            n_done = sum(1 for r in finished_results if r is not None)
            elapsed = time.time() - start_time
            print(
                f"[brawl] ply {ply}: {n_done}/{num_games} games done ({elapsed:.1f}s)", flush=True
            )

    elapsed = time.time() - start_time
    # Translate per-game results to per-engine score.
    a_score = b_score = 0.0
    a_w = a_d = a_l = b_w = b_d = b_l = 0
    for i, r in enumerate(finished_results):
        white_is_a = engine_for_white[i] == engine_a
        if r == "W":
            if white_is_a:
                a_score += 1.0
                a_w += 1
                b_l += 1
            else:
                b_score += 1.0
                b_w += 1
                a_l += 1
        elif r == "L":
            if white_is_a:
                b_score += 1.0
                b_w += 1
                a_l += 1
            else:
                a_score += 1.0
                a_w += 1
                b_l += 1
        else:
            a_score += 0.5
            b_score += 0.5
            a_d += 1
            b_d += 1

    # Write all games to a single PGN file for inspection.
    pgn_written: Path | None = None
    if pgn_path is not None:
        pgn_path.parent.mkdir(parents=True, exist_ok=True)
        a_name = Path(engine_a).stem
        b_name = Path(engine_b).stem
        with pgn_path.open("w", encoding="utf-8") as fh:
            for i, hist in enumerate(histories):
                # Replay history to build the chess.pgn.Game with proper SAN.
                white_is_a = engine_for_white[i] == engine_a
                game = pgn.Game()
                game.headers["Event"] = "MCTS brawl"
                game.headers["Site"] = "local"
                game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                game.headers["Round"] = str(i + 1)
                game.headers["White"] = a_name if white_is_a else b_name
                game.headers["Black"] = b_name if white_is_a else a_name
                r = finished_results[i]
                if r == "W":
                    game.headers["Result"] = "1-0"
                elif r == "L":
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"
                node = game
                replay = chess.Board()
                for u in hist:
                    try:
                        m = chess.Move.from_uci(u)
                        if m in replay.legal_moves:
                            node = node.add_variation(m)
                            replay.push(m)
                        else:
                            break
                    except ValueError:
                        break
                exporter = pgn.FileExporter(fh)
                game.accept(exporter)
                fh.write("\n")
        pgn_written = pgn_path

    return {
        "engine_a": engine_a,
        "engine_b": engine_b,
        "num_games": num_games,
        "ply_count": ply,
        "elapsed_s": elapsed,
        "a_score": a_score,
        "b_score": b_score,
        "a_wdl": (a_w, a_d, a_l),
        "b_wdl": (b_w, b_d, b_l),
        "results_per_color": Counter(
            f"{'A' if engine_for_white[i] == engine_a else 'B'}-{r}"
            for i, r in enumerate(finished_results)
        ),
        "pgn_path": str(pgn_written) if pgn_written else None,
    }


def _autopick_engines(run_dir: Path, lookback: int) -> tuple[Path, Path]:
    engines_dir = run_dir / "engines"
    if not engines_dir.is_dir():
        raise FileNotFoundError(f"no engines/ dir in {run_dir}")
    candidates = sorted(engines_dir.glob("step_*.trt"))
    if len(candidates) < 2:
        raise RuntimeError(f"need >=2 step_*.trt engines; have {len(candidates)}")
    latest = candidates[-1]
    latest_n = int(latest.name.split("_")[-1].removesuffix(".trt"))
    target = max(0, latest_n - lookback)
    older = min(
        candidates,
        key=lambda p: abs(int(p.name.split("_")[-1].removesuffix(".trt")) - target),
    )
    return latest, older


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine-a", type=Path, help="newer / first engine .trt")
    p.add_argument("--engine-b", type=Path, help="older / second engine .trt")
    p.add_argument("--run-dir", type=Path, help="run dir; auto-pick latest vs latest-lookback")
    p.add_argument(
        "--lookback",
        type=int,
        default=500,
        help="if --run-dir given, brawl latest vs latest minus this many steps",
    )
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--simulations", type=int, default=600)
    p.add_argument("--cpuct", type=float, default=0.65)
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="non-root temperature (rarely matters with no-dirichlet)",
    )
    p.add_argument(
        "--temp-root",
        type=float,
        default=0.2,
        help="root temperature; lower = more deterministic move pick",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pgn",
        type=Path,
        default=None,
        help="Write all games as PGN to this path. Default: "
        "artifacts/pgns/mcts_brawl_<timestamp>.pgn. "
        "Pass empty string to disable.",
    )
    args = p.parse_args()

    if args.engine_a and args.engine_b:
        engine_a, engine_b = args.engine_a, args.engine_b
    elif args.run_dir:
        engine_a, engine_b = _autopick_engines(args.run_dir, args.lookback)
        print(
            f"[brawl] auto: A={engine_a.name} (latest)  B={engine_b.name} (latest - ~{args.lookback})",
            flush=True,
        )
    else:
        print("Provide --engine-a + --engine-b OR --run-dir", file=sys.stderr)
        return 2

    if not engine_a.is_file() or not engine_b.is_file():
        print(
            f"engine missing: {engine_a if not engine_a.is_file() else engine_b}", file=sys.stderr
        )
        return 2

    if args.pgn is not None:
        pgn_path = args.pgn if str(args.pgn) else None
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pgn_path = Path(f"artifacts/pgns/mcts_brawl_{ts}.pgn")

    result = play_brawl(
        str(engine_a),
        str(engine_b),
        num_games=args.num_games,
        simulations=args.simulations,
        cpuct=args.cpuct,
        temp=args.temperature,
        temp_root=args.temp_root,
        seed=args.seed,
        pgn_path=pgn_path,
    )

    a_w, a_d, a_l = result["a_wdl"]
    b_w, b_d, b_l = result["b_wdl"]
    n = result["num_games"]
    a_score = result["a_score"]
    b_score = result["b_score"]
    print()
    print(f"=== {n} games in {result['elapsed_s']:.1f}s ({result['ply_count']} plies max) ===")
    print(f"A = {Path(result['engine_a']).name}")
    print(f"B = {Path(result['engine_b']).name}")
    print(f"  A: W={a_w}  D={a_d}  L={a_l}   score={a_score}/{n} ({100 * a_score / n:.1f}%)")
    print(f"  B: W={b_w}  D={b_d}  L={b_l}   score={b_score}/{n} ({100 * b_score / n:.1f}%)")
    print()
    print("split by color:")
    for k in sorted(result["results_per_color"]):
        print(f"  {k}: {result['results_per_color'][k]}")
    if result.get("pgn_path"):
        print()
        print(f"PGN: {result['pgn_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
