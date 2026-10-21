# Next round: faster, cleaner self-play learning

Started 2026-10-21. Budget: **$200 of rented GPU time** (Vast.ai), tracked by
`scripts/vast/budget_guard.py`. This file is the plan and the lab notebook —
results get appended to the log at the bottom as they land.

## Why (audit of `main` @ 128893c)

| Finding | Where | Consequence |
|---|---|---|
| Policy target is the one move *sampled* at τ=1; the visit distribution is discarded | `cpp/selfplay.cpp` (`record.policy = summary.action`) | one noisy sample per position instead of ~30 visit fractions |
| Root τ=1 for the whole game, no resign, 600-ply cap | `config.yaml`, `cpp/selfplay.cpp` | game outcome `z` reflects sampling noise; z/q agree ~67% |
| Policy loss weight 0.1 vs 1.0 + 1.0 for the value heads | `config.yaml` | policy starved of gradient |
| Random-replacement 2M buffer + 200k random-net warmup | `cpp/selfplay.cpp`, `config.yaml` | stale/garbage data lingers; slow start |
| Trunk is ~0.36M params (4×64); policy FC is ~0.76M | `src/models/small/small_model.py` | capacity is tiny; GPU compute is ~2% of inference latency, so growth is cheap |
| Absolute board orientation | `src/models/small/fen_encoder.py` | net learns both colour perspectives (speedrun measured ×1.53 for canonical) |
| Python per-sample batch build, full-buffer re-parse every 20 s, synchronous TRT build | `src/train/` | trainer stalls; becomes the bottleneck once self-play speeds up |
| 8 unused heavy deps (torchtune, torchao, torch-tensorrt, modelopt, onnx-graphsurgeon, onnxruntime, pycuda, onnxscript) | `pyproject.toml` | slow installs on rented boxes |

## Plan

Every new behaviour is behind a config flag whose "off" value reproduces the
old behaviour, so A/B arms run the same binary.

1. **Signal fixes** — record v2 (current bitboards + top-K visit distribution),
   soft policy loss with equal weights, τ cutoff after N plies, resignation with
   a no-resign fraction, playout-cap randomization, FIFO buffer with a growing
   sampling window, vectorized loader, async engine build.
2. **Growth** — function-preserving depth/width growth of the ResNet trunk
   (zero-init residual blocks), triggered on plateau or schedule.
3. **Encoder** — a *small* speedrun-style encoder (canonical board, x0
   re-injection, QK-norm, geometric attention bias, attention policy head),
   sized to the self-play budget (~1–3M params), paired with fewer sims.
4. **Hyperparameter search** — successive halving over short fixed-budget runs,
   scored against a fixed Stockfish ladder + puzzles (not the self-relative chain).
5. **Final long run** with the winning recipe.

## Budget ($200)

Rented boxes: RTX 4090 / 5090 with ≥32 effective CPU cores (MCTS is CPU-bound);
~$0.55/h for a 4090 on 2026-10-21. H100/H200 are a poor fit (~$3.4–4/h, ≤28 cores).

| Phase | Boxes × hours | Est. |
|---|---|---:|
| Bring-up, build, parity tests, size/throughput bench | 1 × 16 | $9 |
| A/B: baseline vs signal fixes | 2 × 24 | $27 |
| Growth + encoder arms | 3 × 24 | $40 |
| HP search (successive halving) | ~6×8 + 3×8 | $40 |
| Final run | 1 × 96 | $55 |
| Reserve | | $29 |
| **Total** | | **$200** |

The guard refuses new launches above $180 committed and destroys every
instance at $200.

## Measurement protocol (for the write-up)

- **Arms.** A = `main@128893c` untouched (`box.sh main-setup` / `main-run`);
  B = `configs/main.yaml` (main's recipe on the new infrastructure);
  C = `configs/signal.yaml` (B + learning-signal fixes); later D = growth,
  E = encoder, then the tuned recipe.
- **Hardware.** One single-GPU box per arm (RTX 4090/5090, >=32 cores). The
  speedup is reported as two factors measured separately: learning efficiency
  (Elo vs self-play positions, hardware independent) and throughput
  (positions per hour; `scripts/bench_sizes.py` on one box + per-arm logs).
- **Strength.** Anchor ladder: random mover, greedy capture, node-limited
  Stockfish at several skills plus skill-0 Stockfish mixed with random moves —
  node limits keep it CPU-speed independent. Anchors are calibrated once by a
  round robin (random = 0, `src/rating/anchors.json`). Every 30 min (same
  cadence in every arm) the newest engine plays 48 paired-opening games
  (MCTS-100) against the two anchors nearest its last estimate; Elo is the MLE
  against fixed anchor ratings, with a standard error. Plus strict-line
  accuracy on 1,000 Lichess puzzles (200 per rating band).
- **Headline.** Time (and positions) to first reach fixed ladder thresholds,
  per arm, with uncertainty from the ladder standard errors. Raw JSONL for
  every chart is synced home to `~/chessrl-runs/<box>/`.

## Log

- 2026-10-21 — audit done, branch `worktree-next-round` created.
- 2026-10-21 — dev box `dev1` (RTX 4090, 32 cores, $0.61/h). C++ built first
  try on TensorRT 10.14 (apt now defaults to TRT 11, pinned back); 69/69
  tests pass on GPU; 3,000 C++-written v2 records validated against
  python-chess (`scripts/check_buffer.py`).
- Bugs found in main while wiring this up:
  1. Every engine swap rebuilt the self-play worker, discarding all 1,024
     games in progress (records are only flushed at game end) — long games
     could never finish. Now `SelfPlayWorker.set_engine()` swaps in place.
  2. Trainer pacing slept ~410 s on its first step (rate estimated from an
     empty window). Now a token bucket: target_k samples per new record.
- Top-20 visit entries kept only 74% of the visit mass (p10 58%) under an
  untrained net — records widened to 84 entries (1,280 bytes).
- Smoke run (signal config, untrained net, 4090): ~450-500 recorded
  positions/s (25% full searches), ~500k nodes/s, 13-16 games/s; policy loss
  6.41 -> 4.29 in 200 steps; in-place engine swap at step 200 confirmed;
  first ladder rating 55 ± 42 Elo, puzzles 6.3% (68 s per rating).
- Anchor calibration: Stockfish skill 0 at 1/10/50 nodes all land within
  ~50 Elo (skill caps depth), so random-move mixtures fill the greedy ->
  Stockfish gap.
