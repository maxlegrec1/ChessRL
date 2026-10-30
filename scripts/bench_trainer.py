"""Where the trainer's step time goes (docs/TRAINER_SPEED.md, "First: measure").

Builds the real Trainer from a config (resumed from a checkpoint, sampling a real self-play
buffer, no network exports), then times:
  data      the batch pipeline alone (Prefetcher: sampling, decoding, pinning), batches/s
  reused    _train_step on one batch already on the GPU: the step's own GPU and launch cost
  fresh     batches from the Prefetcher, copied to the GPU, then _train_step (as run() does)
  profile   torch.profiler over a few reused steps: GPU time per step, kernels per step, and
            the operators that take the most GPU time

    uv run python scripts/bench_trainer.py --buffer runs/<id>/buffer/buffer.bin \\
        --ckpt runs/<id>/checkpoints/step_00092500.pt [--steps 100] [--out bench.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train.config import load_config  # noqa: E402
from src.train.dataset import Prefetcher  # noqa: E402
from src.train.trainer import Trainer  # noqa: E402


def timed(fn, n: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=ROOT / "src/train/config.yaml")
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--compile", action="store_true", help="torch.compile the model (config compile)")
    a = p.parse_args()

    work = Path(tempfile.mkdtemp(prefix="bench_trainer_"))
    cfg = load_config(a.config)
    cfg.data.dataset_path = a.buffer.resolve()
    cfg.resume_path = a.ckpt.resolve()
    cfg.checkpoint.dir = work / "checkpoints"
    cfg.checkpoint.build_script = None
    cfg.metrics_path = work / "train.jsonl"
    cfg.wandb.enabled = False
    if a.compile:
        cfg.compile = True
    trainer = Trainer(cfg)
    out: dict = {"batch_size": cfg.batch_size, "compile": cfg.compile, "amp": cfg.amp_dtype,
                 "gpu": torch.cuda.get_device_name(), "params": trainer.param_count()}  # fmt: skip

    prefetch = Prefetcher(trainer.buffer, cfg.batch_size, cfg.mirror_prob)
    for _ in range(5):
        prefetch.get()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        prefetch.get()
    out["data_batches_per_s"] = a.steps / (time.perf_counter() - t0)

    batch = prefetch.get().to(trainer.device)
    for _ in range(5):  # warm-up: allocator, cuBLAS heuristics
        trainer._train_step(batch, expensive=False)
    out["reused_step_ms"] = 1000 * timed(
        lambda: trainer._train_step(batch, expensive=False), a.steps
    )

    def fresh() -> None:
        trainer._train_step(prefetch.get().to(trainer.device), expensive=False)

    out["fresh_step_ms"] = 1000 * timed(fresh, a.steps)
    out["expensive_step_ms"] = 1000 * timed(lambda: trainer._train_step(batch, expensive=True), 3)

    n_prof = 10
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(n_prof):
            trainer._train_step(batch, expensive=False)
        torch.cuda.synchronize()

    def dev_time(e) -> float:
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0)

    def on_gpu(e) -> bool:
        return str(getattr(e, "device_type", "")).endswith("CUDA")

    kernels = [e for e in prof.events() if on_gpu(e)]
    total = sum(dev_time(e) for e in kernels) or 1.0
    out["profile_gpu_ms_per_step"] = total / 1000 / n_prof
    out["profile_kernels_per_step"] = len(kernels) / n_prof
    # Kernels by name, by GPU time.
    by_name = sorted((e for e in prof.key_averages() if on_gpu(e)), key=dev_time, reverse=True)
    out["top_kernels"] = [
        {
            "kernel": e.key[:90],
            "gpu_ms_per_step": round(dev_time(e) / 1000 / n_prof, 3),
            "share": round(dev_time(e) / total, 3),
            "calls_per_step": round(e.count / n_prof, 1),
        }  # fmt: skip
        for e in by_name[:30]
    ]
    # The PyTorch operators that launched them (each kernel counted once, under the operator that
    # launched it directly), with their input shapes: says which part of the step costs what.
    by_op = sorted(
        (
            e
            for e in prof.key_averages(group_by_input_shape=True)
            if not on_gpu(e) and dev_time(e) > 0
        ),
        key=dev_time,
        reverse=True,
    )
    out["top_ops"] = [
        {
            "op": e.key,
            "shapes": str(e.input_shapes)[:110],
            "gpu_ms_per_step": round(dev_time(e) / 1000 / n_prof, 3),
            "share": round(dev_time(e) / total, 3),
            "calls_per_step": round(e.count / n_prof, 1),
        }
        for e in by_op[:40]
    ]
    prefetch.close()

    print(f"[bench_trainer] {out['gpu']}, batch {cfg.batch_size}, {out['params']:,} parameters")
    print(f"  data pipeline: {out['data_batches_per_s']:.1f} batches/s")
    print(
        f"  step, reused batch: {out['reused_step_ms']:.1f} ms; fresh batches: {out['fresh_step_ms']:.1f} ms"
    )
    print(f"  GPU time per step (profiler): {out['profile_gpu_ms_per_step']:.1f} ms in "
          f"{out['profile_kernels_per_step']:.0f} kernels")  # fmt: skip
    print("  kernels:")
    for row in out["top_kernels"]:
        print(
            f"    {row['gpu_ms_per_step']:7.2f} ms  {100 * row['share']:5.1f}%  x{row['calls_per_step']:<6} {row['kernel']}"
        )
    print("  operators (GPU time of the kernels each launched, by input shapes):")
    for row in out["top_ops"]:
        print(
            f"    {row['gpu_ms_per_step']:7.2f} ms  {100 * row['share']:5.1f}%  x{row['calls_per_step']:<5} "
            f"{row['op']:<28} {row['shapes']}"
        )
    if a.out:
        a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
