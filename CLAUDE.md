# Claude memory for this repo

## GPUs

This box is a **single-GPU NVIDIA H200** (Hopper, sm_90). There is no
second usable GPU. Training, self-play, brawls, and benchmarks all share
this one card.

For Python jobs:
```
CUDA_VISIBLE_DEVICES=0 uv run python ...
```

The C++ build defaults to `CHESSRL_CUDA_ARCH=sm_90` for this hardware.
On other GPUs override the env var (sm_89 for 4090, sm_86 for 3090,
sm_80 for A100) — see README prerequisites.

## Run-monitoring norms

- Active runs live under `artifacts/runs/<id>/`.
- Trainer writes student checkpoints `step_NNNN.pt` and teacher snapshots
  `teacher_step_NNNN.pt` every 100 training steps.
- **Always benchmark the student** (`step_NNNN.pt`) — that's the model self-play
  exports to TRT. The teacher is currently vestigial (saved but not used; an
  earlier BN-buffer EMA bug made it broken anyway).
- `learning_check.py` defaults to the student. Pass `--teacher` only to inspect
  the (broken) teacher snapshot.

## Resume

- `uv run python -m src.run --resume <run_dir>` resumes from the latest student
  checkpoint. Optimizer + scaler + step are persisted in `step_NNNN.pt`.
- Self-play binary is unchanged across runs; it just reads the existing buffer
  and the latest engine via `engines/.iteration`.

## Self-play buffer

- Fixed-size random-replacement buffer (`max_buffer_size` config field, currently
  1,000,000 records).
- Once at capacity, file size stops growing. **Use `<buffer>.writes` sidecar**
  for insertion rate. The trainer's pacing uses this; the file-size probe is
  for the warmup gate only.
- Records are flushed in BURSTS at game-end (1024 boards × ~80 plies each
  flushed at once). Per-step instantaneous rate is highly variable; the
  trainer absorbs this with a 30-second sliding window over the writes counter.
