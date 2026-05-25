# ChessRL architecture

## Process layout

```
src.run (launcher, ~50 lines)
├── src.self_play       ──┐
└── src.train             │
                          │
                          ▼
       artifacts/runs/<id>/
       ├── buffer/buffer.bin                ← shared mmap, 832-byte records
       ├── engines/step_NNNNNNNN.trt       ← self-play polls these
       ├── engines/.iteration              ← pointer to the active engine
       ├── checkpoints/step_NNNNNNNN.pt    ← trainer writes these
       ├── pgns/                           ← optional self-play PGN dumps
       ├── wandb/                          ← W&B local cache
       ├── wandb_run_id.txt                ← surfaces the run id to external metric producers
       ├── config.yaml                     ← resolved config snapshot
       ├── selfplay.log / training.log
       └── ...
```

The two child processes communicate only through the **buffer file** + the
**iteration pointer**. No shared memory, no IPC. This makes them
independently restartable.

## Self-play loop (C++)

`src.self_play` is a thin Python wrapper around the C++ `_inference_cpp`
extension. The C++ side:

1. Loads the engine pointed to by `engines/.iteration`.
2. Maintains 1024 (`selfplay.boards`) concurrent games.
3. For each game: runs MCTS-800 with virtual-loss-batched descent
   (`virtual_loss_width=8`), collecting ~128 leaves per inference round.
4. Writes per-move records to `buffer/buffer.bin` in a 832-byte format
   (111 planes × 8 × 8 fp16 + policy target + value target + MLH target +
   metadata).
5. Polls `engines/.iteration` on a sim boundary; if it changed, swaps
   engines without restarting the process.

The C++ extension lives under `cpp/` and is built via the uv workspace
member declaration in `pyproject.toml`.

## Trainer (Python)

`src.train`:

1. Memory-maps `buffer/buffer.bin` and reloads its view every
   `data.reload_seconds`.
2. Samples `batch_size=256` records per step, applies symmetry
   augmentations on the fly.
3. Computes the multi-head loss (policy + value + value_q + mlh + mlh_q),
   weighted by `loss_weights`.
4. Steps Adam, logs to W&B every `log_every=50` steps.
5. Saves `step_NNNN.pt` every `checkpoint.every_steps` steps, then
   `subprocess.run`s `scripts/build_trt.sh` to convert it to a TRT engine
   under `engines/`, then updates `engines/.iteration` to point at it.
6. Paces itself: holds the throughput at
   `g_records_per_sec / target_k`, so each self-play record is sampled on
   average `target_k` times before churning out of the buffer.

## Buffer record format (832 bytes)

```
offset  size  field
─────── ────  ─────
   0    777   input planes (111 × 8 × 8 fp16, packed C-order)  
 777      1   stm (side to move, redundant with planes; kept for sanity)
 778      4   policy_target_idx (uint32, vocab index of the MCTS action)
 782      2   value_outcome (W/D/L as fp16 trinary)
 784      2   value_q (MCTS root Q, fp16)
 786      2   mlh_target (plies remaining to end of game, int16; -1 = unknown)
 788     44   reserved / metadata (game id, ply, etc.)
```

Records are append-only within a "burst" (a full self-play game ends → all
its plies flush at once). The buffer is fixed-capacity
(`selfplay.max_buffer_size`); writes beyond capacity randomly replace
existing records.

## Engine swap protocol

```
trainer                                          self_play
───────                                          ─────────
checkpoint saved (step N)
  ↓
scripts/build_trt.sh:
  pt → onnx → trtexec → step_N.trt
  ↓
mv step_N.trt engines/
  ↓
echo "/path/to/engines/step_N.trt" > engines/.iteration  ──→  read on next sim boundary
                                                              swap engine pointer
                                                              continue MCTS
```

No restart. No process synchronization beyond the atomic file rename.

## Resume

`src.run --resume <run_dir>` reads `<run_dir>/config.yaml`, sets
`resume_path` to the latest checkpoint, and starts both children. The
trainer:

1. Loads model weights + optimizer state + scaler state + step counter
   from the checkpoint.
2. Re-attaches to the existing W&B run via `wandb_run_id.txt` if
   `wandb.resume_id` is set.
3. The buffer file is unchanged — training picks up exactly where it
   stopped.

The self-play binary is unchanged across runs; it just reads the
existing buffer and the latest engine via `engines/.iteration`.

## Why 1.29M parameters?

The model is small on purpose. On an H200:

- Per-batch GPU compute ≈ 21 µs out of ~1 ms total latency.
- 97% of inference wall time is launch overhead + H2D/D2H + sync, not math.
- CUDA graphs absorb the launch overhead (~1.8× end-to-end win).
- Throughput is bound by data path bandwidth (111 dense planes per leaf),
  not compute.

A bigger model wouldn't be faster on this hardware until the data path
collapses to packed bitboards (see `archive/legacy_scripts/` for the
packed-refactor investigation that didn't ship).

## Where things live

| Concern | Code |
|---|---|
| Top-level launcher | `src/run.py` |
| Training loop | `src/train/trainer.py` |
| Buffer dataset / record parsing | `src/train/dataset.py` |
| Self-play orchestrator (Python side) | `src/self_play/__init__.py` |
| C++ MCTS + inference | `cpp/mcts/`, `cpp/inference.cpp`, `cpp/selfplay.cpp` |
| Model definitions | `src/models/small/small_model.py`, `src/models/leela/model.py` |
| FEN encoding (111-plane) | `src/models/small/fen_encoder.py` |
| Eval: head-to-head | `src/eval/brawls.py`, `src/eval/mcts_brawl.py` |
| Eval: vs Stockfish | `scripts/stockfish_match.py` |
| Engine build (`.pt` → `.trt`) | `scripts/build_trt.sh` (calls `src/export/to_onnx.py` + `trtexec`) |

For the autobrawl daemon, chain Elo brawler, and dashboard, see the
[scripts table in the README](../README.md#scripts-overview).
