"""Run the SelfPlayWorker for a fixed wall-clock duration and dump a JSON
perf snapshot. Used by the self-play throughput overhaul to establish a
baseline and compare phases.

Default config matches the production self-play knob set: concurrency=1024,
threads=256, sims=800, hash_size=0 (TT off), fp16 engine. The output buffer
file is written into a scratch directory under artifacts/perf/scratch/ so
the production buffer is never touched.

Example:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_selfplay.py \\
      --engine artifacts/legal_ceiling_v7_history_arch/v7.trt \\
      --duration-sec 90 \\
      --out artifacts/perf/phase0.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import subprocess
import sys
import time
from typing import Any

import _inference_cpp as cpp


def gpu_name() -> str:
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()
        )
        if out:
            return out[0]
    except Exception:
        pass
    return "unknown"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path to .trt engine (must be 111-plane).")
    ap.add_argument("--duration-sec", type=float, default=90.0)
    ap.add_argument(
        "--warmup-sec",
        type=float,
        default=10.0,
        help="Initial seconds excluded from the throughput average; lets stream/exec contexts warm up.",
    )
    ap.add_argument("--concurrency", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--hash-size", type=int, default=0)
    ap.add_argument("--max-buffer-size", type=int, default=1_000_000)
    ap.add_argument("--temperature", type=float, default=1.3)
    ap.add_argument("--temperature-at-root", type=float, default=1.0)
    ap.add_argument("--cpuct", type=float, default=0.65)
    ap.add_argument("--use-dirichlet", action="store_true", default=True)
    ap.add_argument("--no-dirichlet", dest="use_dirichlet", action="store_false")
    ap.add_argument("--dirichlet-alpha", type=float, default=0.12)
    ap.add_argument("--dirichlet-epsilon", type=float, default=0.05)
    ap.add_argument("--random-opening-plies", type=int, default=0)
    ap.add_argument(
        "--virtual-loss-width",
        type=int,
        default=1,
        help="MCTS virtual-loss batched-descent width. 1 = legacy single-leaf descent.",
    )
    ap.add_argument("--scratch-dir", default="artifacts/perf/scratch")
    ap.add_argument(
        "--label", default=None, help="Free-form label saved into the JSON (e.g. phase0, phase1)."
    )
    ap.add_argument("--out", required=True, help="Where to write the JSON snapshot.")
    ap.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Leave the scratch buffer on disk after the bench.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not pathlib.Path(args.engine).exists():
        print(f"Engine not found: {args.engine}", file=sys.stderr)
        return 1

    scratch = pathlib.Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    buffer_path = scratch / f"bench_{int(time.time())}.bin"
    if buffer_path.exists():
        buffer_path.unlink()

    label = args.label or pathlib.Path(args.out).stem
    print(f"[bench] engine={args.engine}")
    print(f"[bench] buffer={buffer_path}")
    print(
        f"[bench] config: concurrency={args.concurrency} threads={args.threads} "
        f"sims={args.sims} hash_size={args.hash_size}"
    )
    print(f"[bench] warmup={args.warmup_sec}s measured={args.duration_sec}s")

    worker = cpp.SelfPlayWorker(
        engine_path=args.engine,
        output_path=str(buffer_path),
        batch_size=args.concurrency,
        threads=args.threads,
        hash_size=args.hash_size,
        max_buffer_size=args.max_buffer_size,
        simulations=args.sims,
        cpuct=args.cpuct,
        temperature=args.temperature,
        temperature_at_root=args.temperature_at_root,
        use_dirichlet=args.use_dirichlet,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        random_opening_plies=args.random_opening_plies,
        virtual_loss_width=args.virtual_loss_width,
        max_games=-1,
        save_pgn=False,
    )

    # ---- Warmup ----------------------------------------------------------
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.warmup_sec:
        worker.step()

    # ---- Measured window -------------------------------------------------
    pre_stats = worker.inference_stats()
    pre_positions = 0
    pre_games = 0
    pre_nodes = 0

    t_start = time.monotonic()
    samples: list[dict[str, Any]] = []
    last_sample = t_start
    step_count = 0

    while True:
        m = worker.step()
        pre_positions += m.positions_generated
        pre_games += m.games_finished
        pre_nodes += m.nodes_evaluated
        step_count += 1

        now = time.monotonic()
        if now - last_sample >= 5.0:
            samples.append(
                {
                    "t": now - t_start,
                    "positions": pre_positions,
                    "games": pre_games,
                    "nodes": pre_nodes,
                }
            )
            last_sample = now
        if now - t_start >= args.duration_sec:
            break

    elapsed = time.monotonic() - t_start
    post_stats = worker.inference_stats()

    delta_batches = post_stats.batches - pre_stats.batches
    delta_positions_in_batches = post_stats.positions_in_batches - pre_stats.positions_in_batches
    delta_infer_ms = post_stats.total_inference_ms - pre_stats.total_inference_ms
    delta_encoder_calls = post_stats.encoder_calls - pre_stats.encoder_calls
    delta_encoder_ms = post_stats.total_encoder_ms - pre_stats.total_encoder_ms

    pos_per_sec = pre_positions / elapsed if elapsed > 0 else 0.0
    games_per_sec = pre_games / elapsed if elapsed > 0 else 0.0
    nodes_per_sec = pre_nodes / elapsed if elapsed > 0 else 0.0
    mean_batch_size = (delta_positions_in_batches / delta_batches) if delta_batches else 0.0
    mean_infer_ms_per_batch = (delta_infer_ms / delta_batches) if delta_batches else 0.0
    mean_encode_us_per_leaf = (
        (delta_encoder_ms * 1000.0 / delta_encoder_calls) if delta_encoder_calls else 0.0
    )
    # Encoder time charged to N MCTS worker threads in parallel; per-leaf wall-clock
    # is the relevant number for hot-path cost, while the *aggregate* time gives the
    # fraction of total wall-clock the threads spent encoding.
    encode_thread_busy_frac = delta_encoder_ms / 1000.0 / max(1.0, elapsed * args.threads)
    leaves_per_batch_target = delta_encoder_calls / delta_batches if delta_batches else 0.0

    print()
    print(
        f"[bench] {label}: {pos_per_sec:.1f} positions/sec  "
        f"({games_per_sec:.2f} games/sec, {nodes_per_sec:.0f} nodes/sec)"
    )
    print(
        f"[bench]   {step_count} steps in {elapsed:.1f}s, {pre_games} games, {pre_positions} positions"
    )
    print(
        f"[bench]   inference: {delta_batches} batches, avg {mean_batch_size:.1f}/batch, "
        f"{mean_infer_ms_per_batch:.2f} ms/batch"
    )
    print(
        f"[bench]   encoder:  {delta_encoder_calls} calls, {mean_encode_us_per_leaf:.2f} us/leaf, "
        f"thread-busy frac {encode_thread_busy_frac * 100:.1f}%"
    )

    out = {
        "label": label,
        "engine": args.engine,
        "gpu": gpu_name(),
        "host": socket.gethostname(),
        "ts": time.time(),
        "config": {
            "concurrency": args.concurrency,
            "threads": args.threads,
            "sims": args.sims,
            "hash_size": args.hash_size,
            "max_buffer_size": args.max_buffer_size,
            "temperature": args.temperature,
            "temperature_at_root": args.temperature_at_root,
            "cpuct": args.cpuct,
            "use_dirichlet": args.use_dirichlet,
            "dirichlet_alpha": args.dirichlet_alpha,
            "dirichlet_epsilon": args.dirichlet_epsilon,
            "random_opening_plies": args.random_opening_plies,
        },
        "wall_clock_sec": elapsed,
        "warmup_sec": args.warmup_sec,
        "steps": step_count,
        "positions": pre_positions,
        "games": pre_games,
        "nodes": pre_nodes,
        "throughput": {
            "positions_per_sec": pos_per_sec,
            "games_per_sec": games_per_sec,
            "nodes_per_sec": nodes_per_sec,
        },
        "inference": {
            "batches": delta_batches,
            "positions_in_batches": delta_positions_in_batches,
            "total_inference_ms": delta_infer_ms,
            "mean_batch_size": mean_batch_size,
            "mean_infer_ms_per_batch": mean_infer_ms_per_batch,
        },
        "encoder": {
            "calls": delta_encoder_calls,
            "total_encoder_ms_aggregate": delta_encoder_ms,
            "mean_us_per_leaf": mean_encode_us_per_leaf,
            "leaves_per_batch": leaves_per_batch_target,
            "thread_busy_frac": encode_thread_busy_frac,
        },
        "samples": samples,
        "buffer_path": str(buffer_path),
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {out_path}")

    if not args.keep_scratch and buffer_path.exists():
        buffer_path.unlink()
        sidecar = buffer_path.with_suffix(buffer_path.suffix + ".writes")
        if sidecar.exists():
            sidecar.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
