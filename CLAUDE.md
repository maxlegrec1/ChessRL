# Claude memory for this repo

## GPUs

Everything runs on one GPU: training, self-play, the ladder and benchmarks share it.

```
CUDA_VISIBLE_DEVICES=0 uv run python ...
```

The C++ build compiles for `CHESSRL_CUDA_ARCH` (default `sm_90`, H100/H200). Set it for the
card in use: `sm_89` RTX 4090, `sm_86` RTX 3090, `sm_80` A100 (see the README).

The home box (maxime-mega) has an RTX 4090 but no CUDA toolkit or TensorRT, so the extension
does not build there. Training runs on rented Vast.ai boxes (`scripts/vast/`); their run
directories sync to `~/chessrl-runs/`, which the home dashboard and analysis board read. Ask
before starting anything that costs money.

## Runs

- A run lives in `artifacts/runs/<id>/`: `config.yaml` (the resolved config), `buffer/`,
  `engines/` (+ `.iteration`), `checkpoints/`, `metrics/*.jsonl`, `ladder/`, logs.
- The default config `src/train/config.yaml` is the recipe. Every key is explicit: a missing
  key falls back to the dataclass default in `src/train/config.py`, which is the original
  (first post) recipe.
- The trainer saves `checkpoints/step_NNNNNNNN.pt` every `checkpoint.every_steps` (500) and
  builds an engine from it; self-play swaps engines in place.
- Strength: `scripts/ladder_daemon.py --run-dir <run>` (ladder Elo + puzzles, every 30 min).
  `scripts/trt_parity.py` checks an engine against PyTorch.

## Resume

- `uv run python -m src.run --resume <run_dir>` resumes from the latest checkpoint with the
  run's own `config.yaml`. Optimizer, scaler and step are saved in `step_NNNNNNNN.pt`.
- Self-play reads the existing buffer and the latest engine from `engines/.iteration`.

## Self-play buffer

- A ring buffer (`selfplay.fifo_buffer: true`) of `max_buffer_size` records (2,000,000 x
  1,280 bytes); the trainer samples a window over the newest records (`data.window_*`).
- Once full, the file stops growing. **Use the `<buffer>.writes` sidecar** for the insertion
  count. Trainer pacing is a token bucket on it: at most `target_k` samples per record written
  since training started. The 30-second rate window is only for the logs.
- Records arrive in bursts at game end, and only full searches (25% of moves) are recorded.
