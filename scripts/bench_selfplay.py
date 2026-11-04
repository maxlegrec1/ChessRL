"""Run the SelfPlayWorker for a fixed wall-clock duration and dump a JSON
perf snapshot (compare snapshots with scripts/perfreport.py).

The worker is built exactly as self-play builds it, from the selfplay section of
--config (default: the training recipe); --concurrency/--threads/--sims/
--virtual-loss-width/--hash-size override single knobs for sweeps. Records go to
a scratch buffer under artifacts/bench/, so a run's buffer is never touched.

Example:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/bench_selfplay.py \\
      --engine artifacts/runs/<id>/engines/step_00010000.cnet \\
      --duration-sec 90 --out artifacts/bench/baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
from typing import Any

import _inference_cpp as cpp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.self_play.__main__ import _CPP_FIELDS, build_cpp_config  # noqa: E402
from src.train.config import load_config  # noqa: E402

# Command-line overrides: flag -> SelfPlayConfig field.
OVERRIDES = {
    "concurrency": "boards",
    "threads": "threads",
    "async_play": "async_play",
    "sims": "simulations",
    "virtual_loss_width": "virtual_loss_width",
    "hash_size": "hash_size",
}


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
    ap.add_argument(
        "--engine", required=True, help="Path to the network (.cnet, src/export/to_cnet.py)."
    )
    ap.add_argument("--duration-sec", type=float, default=90.0)
    ap.add_argument(
        "--warmup-sec",
        type=float,
        default=10.0,
        help="Initial seconds excluded from the throughput average; lets stream/exec contexts warm up.",
    )
    ap.add_argument(
        "--config",
        type=pathlib.Path,
        default=ROOT / "src" / "train" / "config.yaml",
        help="Config whose selfplay section the worker uses.",
    )
    for flag in OVERRIDES:
        ap.add_argument(f"--{flag.replace('_', '-')}", type=int, default=None)
    ap.add_argument("--scratch-dir", default="artifacts/bench")
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


def host_busy_sec() -> float:
    """Busy CPU seconds of the whole host since boot (/proc/stat is not per container)."""
    with open("/proc/stat") as fh:
        f = [float(x) for x in fh.readline().split()[1:9]]
    return (sum(f) - f[3] - f[4]) / os.sysconf("SC_CLK_TCK")


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

    sp = load_config(args.config).selfplay
    for flag, name in OVERRIDES.items():
        if getattr(args, flag) is not None:
            setattr(sp, name, type(getattr(sp, name))(getattr(args, flag)))  # 0/1 for a bool key
    sp.save_pgn = False
    settings = {"boards": sp.boards, **{name: getattr(sp, name) for name in _CPP_FIELDS}}

    label = args.label or pathlib.Path(args.out).stem
    print(f"[bench] engine={args.engine}")
    print(f"[bench] buffer={buffer_path}")
    print(
        f"[bench] config: {args.config} boards={sp.boards} threads={sp.threads} "
        f"sims={sp.simulations}/{sp.pcr_fast_simulations} vlw={sp.virtual_loss_width}"
    )
    print(f"[bench] warmup={args.warmup_sec}s measured={args.duration_sec}s")

    worker = cpp.SelfPlayWorker(build_cpp_config(sp, args.engine, buffer_path))

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
    cpu_start = time.process_time()  # every thread of this process: the search and the runner
    host_start = host_busy_sec()
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
    cpu_sec = time.process_time() - cpu_start
    host_load = os.getloadavg()[0]  # in a container: the whole host, neighbours included
    others_sec = host_busy_sec() - host_start - cpu_sec  # CPU anything else on the host used
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
    encode_thread_busy_frac = delta_encoder_ms / 1000.0 / max(1.0, elapsed * sp.threads)
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
    print(
        f"[bench]   cpu: {cpu_sec / elapsed:.1f} cores, "
        f"{cpu_sec * 1000 / max(1, pre_positions):.1f} ms per recorded position, "
        f"host load {host_load:.0f}, rest of the host {others_sec / elapsed:.1f} cores"
    )

    out = {
        "label": label,
        "engine": args.engine,
        "gpu": gpu_name(),
        "host": socket.gethostname(),
        "ts": time.time(),
        "config_file": str(args.config),
        "config": settings,
        "wall_clock_sec": elapsed,
        "cpu_sec": cpu_sec,
        "host_load": host_load,
        "host_others_sec": others_sec,
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
