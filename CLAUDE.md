# Claude memory for this repo

## GPUs

This box has two GPUs but **only one is usable**:

- `nvidia-smi` index **1**: NVIDIA RTX 4090 (24 GB) — the only working GPU. Shows up as
  `CUDA_VISIBLE_DEVICES=0` to programs (the device list is filtered to it).
- `nvidia-smi` index **0**: NVIDIA GTX 1070 (8 GB) — too old for our PyTorch build
  (sm_61, current PyTorch needs sm_75+). **Do not target it.** It will appear "free"
  in `nvidia-smi` but it can't run inference.

When the user says "GPU0" they mean the 4090 (the CUDA-visible one), not what
`nvidia-smi` shows in row 0. Both training, self-play, and any side jobs (brawls,
benchmarks) run on the same physical 4090. There is no "spare GPU" to offload to.

For Python jobs we always use:
```
CUDA_VISIBLE_DEVICES=0 uv run python ...
```
which maps to the 4090.

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
