"""Elo chain brawl: consecutive 5k-step checkpoints play 50 batched MCTS-800
games each (8-random-ply openings, temp_at_root=0, no dirichlet, width=1).
Cross-pair parallelism via multiprocessing.Pool (spawn, CUDA-safe).
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NUM_GAMES = 50
OPENING_PLIES = 8
MAX_PLIES = 400
SIMULATIONS = 800
CPUCT = 0.65
VLW = 1

WANDB_PROJECT = "self_play"  # mirrors src/train/config.py default


def _push_chain_to_wandb(run_dir: Path, rows: list[tuple[int, int, float, float]]) -> None:
    """Attach to the trainer's W&B run and log per-interval ΔElo + cumulative.

    `rows` is a list of (from_step, to_step, delta_elo, cumulative_elo). Each
    row is logged with W&B step = to_step, so the curve aligns with the
    trainer's own step axis. Fail-soft: silent skip if wandb_run_id.txt is
    missing or wandb is unreachable.
    """
    rid_file = run_dir / "wandb_run_id.txt"
    if not rid_file.is_file():
        return
    rid = rid_file.read_text().strip()
    try:
        import wandb

        wandb.init(
            project=WANDB_PROJECT,
            id=rid,
            resume="must",
            reinit=True,
            settings=wandb.Settings(silent=True),
        )
        for from_step, to_step, delta_elo, cum_elo in rows:
            wandb.log(
                {
                    "eval/elo_delta_per_5k": delta_elo,
                    "eval/elo_cumulative": cum_elo,
                    "eval/elo_chain_from_step": from_step,
                },
                step=to_step,
            )
        wandb.finish()
        print(f"[wandb] logged {len(rows)} chain intervals", flush=True)
    except Exception as exc:
        print(f"[wandb] log failed (non-fatal): {exc}", flush=True)


def _brawl_one_pair(engine_a: str, engine_b: str, seed: int):
    """50 batched games of engine_a vs engine_b. Returns (w, d, l) from A's POV."""
    import _inference_cpp  # subprocess-local import

    rng = random.Random(seed)
    boards = [chess.Board() for _ in range(NUM_GAMES)]
    histories = [[] for _ in range(NUM_GAMES)]
    a_is_white = [(i % 2 == 0) for i in range(NUM_GAMES)]
    finished = [False] * NUM_GAMES
    a_score = [0.0] * NUM_GAMES

    # 8 random opening plies, uniform legal sampling (per-game RNG decoupled)
    for _ in range(OPENING_PLIES):
        for i, b in enumerate(boards):
            if b.is_game_over(claim_draw=True):
                continue
            mv = rng.choice(list(b.legal_moves))
            histories[i].append(mv.uci())
            b.push(mv)

    def _settle(i):
        b = boards[i]
        outcome = b.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            a_score[i] = 0.5
        else:
            a_won = (outcome.winner == chess.WHITE) == a_is_white[i]
            a_score[i] = 1.0 if a_won else 0.0
        finished[i] = True

    while not all(finished):
        active_a, active_b = [], []
        for i, b in enumerate(boards):
            if finished[i]:
                continue
            if len(histories[i]) >= MAX_PLIES:
                a_score[i] = 0.5
                finished[i] = True
                continue
            if b.is_game_over(claim_draw=True):
                _settle(i)
                continue
            stm_white = b.turn == chess.WHITE
            is_a_turn = stm_white == a_is_white[i]
            (active_a if is_a_turn else active_b).append(i)

        for engine_path, active in ((engine_a, active_a), (engine_b, active_b)):
            if not active:
                continue
            res = _inference_cpp.small_mcts_search_batch_with_history(
                histories=[histories[i] for i in active],
                fens=["" for _ in active],
                simulations=SIMULATIONS,
                cpuct=CPUCT,
                temperature=1.0,
                engine_path=engine_path,
                temperature_at_root=0.0,
                use_dirichlet=False,
                virtual_loss_width=VLW,
            )
            for k, i in enumerate(active):
                uci = res[k]["action"]
                try:
                    mv = chess.Move.from_uci(uci)
                except Exception:
                    a_lost = engine_path == engine_a
                    a_score[i] = 0.0 if a_lost else 1.0
                    finished[i] = True
                    continue
                if mv not in boards[i].legal_moves:
                    a_lost = engine_path == engine_a
                    a_score[i] = 0.0 if a_lost else 1.0
                    finished[i] = True
                    continue
                histories[i].append(uci)
                boards[i].push(mv)

    w = sum(1 for s in a_score if s == 1.0)
    d = sum(1 for s in a_score if s == 0.5)
    l = sum(1 for s in a_score if s == 0.0)
    return w, d, l


def _elo_diff(score: float, n: int) -> float:
    """ΔElo of A relative to B from a head-to-head score in [0, n]."""
    if score <= 0:
        return -800.0
    if score >= n:
        return 800.0
    p = score / n
    return -400.0 * math.log10((1 - p) / p)


def _worker(args):
    engine_a, engine_b, seed, pair_idx, total_pairs = args
    t0 = time.time()
    try:
        w, d, l = _brawl_one_pair(engine_a, engine_b, seed)
    except Exception as e:
        return ("ERROR", str(e), engine_a, engine_b, pair_idx)
    dt = time.time() - t0
    n = w + d + l
    score = w + 0.5 * d
    elo = _elo_diff(score, n)
    return ("OK", engine_a, engine_b, w, d, l, score, elo, dt, pair_idx, total_pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="artifacts/runs/20260515_133813_99ec_history_fix_v1")
    p.add_argument("--step-bin", type=int, default=5000)
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument("--min-step", type=int, default=None)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default="artifacts/exp/elo_chain")
    args = p.parse_args()

    run_dir = ROOT / args.run_dir
    engines_dir = run_dir / "engines"
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    chain = []
    for e in sorted(engines_dir.glob("step_*.trt")):
        try:
            s = int(e.stem.split("_")[1])
        except Exception:
            continue
        if (
            s % args.step_bin == 0
            and (args.max_step is None or s <= args.max_step)
            and (args.min_step is None or s >= args.min_step)
        ):
            chain.append((s, e))
    if len(chain) < 2:
        print("need ≥2 chain checkpoints")
        return 1

    pairs = [
        (str(chain[i][1]), str(chain[i + 1][1]), i, i, len(chain) - 1)
        for i in range(len(chain) - 1)
    ]
    print(f"chain: {len(chain)} engines from step {chain[0][0]:,} to {chain[-1][0]:,}")
    print(f"pairs: {len(pairs)}   games/pair: {NUM_GAMES}   workers: {args.workers}")
    print(
        f"MCTS: sims={SIMULATIONS} cpuct={CPUCT} vlw={VLW} temp_root=0 "
        f"no-dirichlet  opening_plies={OPENING_PLIES}"
    )
    print()

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(
            [
                "from_step",
                "to_step",
                "from_engine",
                "to_engine",
                "from_w",
                "draws",
                "from_l",
                "from_score",
                "delta_elo",
                "seconds",
            ]
        )

    ctx = mp.get_context("spawn")
    t_chain = time.time()
    done = 0
    with ctx.Pool(args.workers) as pool:
        for res in pool.imap_unordered(_worker, pairs):
            done += 1
            if res[0] == "ERROR":
                _, err, ea, eb, idx = res
                print(f"  [pair {idx}] ERROR  {Path(ea).stem} → {Path(eb).stem}: {err}")
                continue
            _, ea, eb, w, d, l, score, elo, dt, pair_idx, total = res
            from_step = int(Path(ea).stem.split("_")[1])
            to_step = int(Path(eb).stem.split("_")[1])
            print(
                f"  [{done:>2}/{len(pairs)}] {from_step // 1000:>3}k→{to_step // 1000:>3}k  "
                f"score={score:5.1f}/{w + d + l}  (W{w}/D{d}/L{l})  "
                f"ΔElo={elo:+6.0f}  ({dt:.0f}s)",
                flush=True,
            )
            with csv_path.open("a", newline="") as fcsv:
                wr = csv.writer(fcsv)
                wr.writerow([from_step, to_step, ea, eb, w, d, l, score, f"{elo:.1f}", f"{dt:.0f}"])

    elapsed_total = time.time() - t_chain
    print(f"\nchain complete in {elapsed_total / 60:.1f} min")

    # Read results, sort, compute cumulative Elo
    with csv_path.open() as fcsv:
        rows = list(csv.DictReader(fcsv))
    rows.sort(key=lambda r: int(r["from_step"]))
    cum_steps = [int(rows[0]["from_step"])]
    cum_elos = [0.0]
    cum = 0.0
    wandb_rows: list[tuple[int, int, float, float]] = []
    for r in rows:
        cum += float(r["delta_elo"])
        cum_steps.append(int(r["to_step"]))
        cum_elos.append(cum)
        wandb_rows.append((int(r["from_step"]), int(r["to_step"]), float(r["delta_elo"]), cum))

    # Optionally push to the trainer's W&B run.
    _push_chain_to_wandb(run_dir, wandb_rows)

    # Plot per-interval and cumulative
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    labels = [f"{int(r['from_step']) // 1000}k→{int(r['to_step']) // 1000}k" for r in rows]
    deltas = [float(r["delta_elo"]) for r in rows]
    axes[0].bar(labels, deltas, color=["seagreen" if d > 0 else "indianred" for d in deltas])
    axes[0].set_ylabel("ΔElo per 5k-step interval")
    axes[0].set_title(
        f"Per-interval Elo gain  ({NUM_GAMES} games/pair, MCTS-{SIMULATIONS}, 8 random opening plies)"
    )
    axes[0].axhline(0, color="grey", lw=1)
    axes[0].grid(alpha=0.3)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=8)

    axes[1].plot(cum_steps, cum_elos, marker="o", color="darkblue")
    axes[1].fill_between(cum_steps, cum_elos, 0, alpha=0.15, color="darkblue")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel(f"cumulative Elo (relative to {chain[0][0] // 1000}k)")
    axes[1].set_title("Cumulative Elo trajectory")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="grey", lw=1)
    plt.tight_layout()
    png = out_dir / "elo_chain.png"
    plt.savefig(png, dpi=120)
    print(f"CSV  : {csv_path}")
    print(f"PLOT : {png}")


if __name__ == "__main__":
    main()
