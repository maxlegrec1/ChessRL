# Claude memory for this repo

## GPUs

Everything runs on one GPU: training, self-play, the ladder and benchmarks share it.

```
CUDA_VISIBLE_DEVICES=0 uv run python ...
```

The C++ build compiles for `CHESSRL_CUDA_ARCH` (default `sm_90`, H100/H200). Set it for the
card in use: `sm_89` RTX 4090, `sm_86` RTX 3090, `sm_80` A100 (see the README).

The home box (maxime-mega) has an RTX 4090 and, since 2026-11-05, Ubuntu's CUDA 12.4 toolkit (`/usr`) with GCC 13
for nvcc (the system GCC 15 is too new for it). Build there with
`CUDA_ROOT=/usr CHESSRL_CUDA_ARCH=sm_89 NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13" uv sync --reinstall-package
chessrl-inference-cpp`; the extension then loads PyTorch's CUDA 12.8 runtime and cuBLAS (its run path,
`cpp/setup.py`). Rented Vast.ai boxes (`scripts/vast/`) also run training; their run directories sync to
`~/chessrl-runs/`, which the home dashboard and analysis board read. Ask before starting anything that costs money.

Networks wider than the fused kernel holds (d_model 128 / 8 heads / d_ff 256) run on the wide path
(`cpp/net/encoder_wide.cu`, cuBLAS; to_cnet writes them row-major with a `wide` entry), FP16 only. INT8 networks of
d_model 512 / 8 heads / d_ff 768 run their own fused kernel (`cpp/net/encoder_fused512.cu`, a `fused512` entry,
~2.3x the wide path); their INT8 rounds the attention output per head (`src/models/quant.py` int8_groups).

## Runs

- A run lives in `artifacts/runs/<id>/`: `config.yaml` (the resolved config), `buffer/`,
  `engines/` (+ `.iteration`), `checkpoints/`, `metrics/*.jsonl`, `ladder/`, logs.
- The default config `src/train/config.yaml` is the recipe, with every setting it uses
  written out: a missing key falls back to the dataclass default in `src/train/config.py`
  (older values, not the recipe).
- The trainer saves `checkpoints/step_NNNNNNNN.pt` every `checkpoint.every_steps` (500) and
  exports it to `engines/step_NNNNNNNN.cnet` (`scripts/build_net.sh`, the fused CUDA network in
  `cpp/net/`); self-play swaps networks in place.
- Strength: `scripts/ladder_daemon.py --run-dir <run>` (ladder Elo + puzzles, every 30 min).
  `scripts/net_parity.py` checks the fused network against PyTorch.

## Resume

- `uv run python -m src.run --resume <run_dir>` resumes from the latest checkpoint with the
  run's own `config.yaml`. Optimizer, scaler and step are saved in `step_NNNNNNNN.pt`.
- Self-play reads the existing buffer and the latest engine from `engines/.iteration`.

## Self-play buffer

- A ring buffer (`selfplay.fifo_buffer: true`) of `max_buffer_size` records (6,000,000 x
  1,280 bytes = 7.7 GB); the trainer samples a window over the newest records (`data.window_*`).
- A run resumed into an empty buffer trains on a tiny window and overfits for ~1 h (−150 Elo or
  more): branch runs should copy their parent's `buffer.bin*` (records + `.writes`).
- Once full, the file stops growing. **Use the `<buffer>.writes` sidecar** for the insertion
  count. Trainer pacing is a token bucket on it: at most `target_k` samples per record written
  since training started. The 30-second rate window is only for the logs.
- Records arrive in bursts at game end, and only full searches (25% of moves) are recorded.
