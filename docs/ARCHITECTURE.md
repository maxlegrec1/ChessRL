# ChessRL architecture

How the code works: the processes and the files they share, the search, the trainer, the record
format, the engines and the networks. Numbers in parentheses are the recipe's values
(`src/train/config.yaml`); every field is described in [`CONFIG.md`](CONFIG.md).

## Process layout

`uv run python -m src.run` (`src/run.py`) is the launcher. For a new run it creates
`artifacts/runs/<YYYYmmdd_HHMMSS>_<4 hex>[_<name>]/`, saves a randomly initialised network as
`checkpoints/model_dynamic_initial.pt` and builds a TensorRT engine from it (so self-play has an
engine from the start), writes the config to `<run>/config.yaml` with every path pointed inside
the run directory, and starts `python -m src.self_play` and `python -m src.train` on that file,
appending their output to `selfplay.log` and `training.log`. Signals are forwarded to both
children; when one exits, the launcher stops the other. `--no-train` or `--no-selfplay` runs one
side only. The two processes share nothing but files:

```
artifacts/runs/<id>/
├── config.yaml                       resolved config; both processes read it
├── buffer/buffer.bin                 records, 1,280 bytes each: self-play writes, the trainer memory-maps
├── buffer/buffer.bin.writes          number of records ever written: pacing and the ring's head
├── buffer/buffer.bin.stats.json      capacity, FIFO flag, cumulative game statistics
├── checkpoints/step_NNNNNNNN.pt      trainer checkpoints
├── engines/step_NNNNNNNN.{onnx,trt}  engines built from them
├── engines/.iteration                CSV of published engines: self-play plays the last row
├── metrics/train.jsonl               trainer metrics and events (growth, skipped builds)
├── metrics/selfplay.jsonl            self-play speed and game statistics, every 30 s
├── ladder/ladder.jsonl               ratings from scripts/ladder_daemon.py
├── selfplay.log, training.log, engine_builds.log
└── pgns/, wandb/                     optional PGNs (selfplay.save_pgn) and W&B files
```

Either process can be stopped and started again on its own (see [Resume](#resume)).

## Self-play

`src/self_play/__main__.py` copies the `selfplay` fields onto the C++ `SelfPlayConfig`
(`_CPP_FIELDS`) and drives `_inference_cpp.SelfPlayWorker` (`cpp/selfplay.cpp`). One call to
`worker.step()` plays one move in each of the `boards` (1,024) games: it runs one search per game
on `threads` (128) CPU threads, applies the moves, ends finished games and writes their records.
Between steps the Python loop looks for a new engine and, every 30 s, appends to
`metrics/selfplay.jsonl`.

**Per-move settings.**

- *Playout-cap randomisation* (KataGo's idea: vary the search size from move to move): a move gets
  a full, recorded search of `simulations` (400) with probability `pcr_full_prob` (0.25), else an
  unrecorded `pcr_fast_simulations` (50) search without noise that only moves the game forward.
- *Temperatures*: the priors are softmax(logits / `temperature`) (1.3) at every node. For the
  first `temperature_plies` (30) searched plies the move is sampled with probability proportional
  to visits^(1/`temperature_at_root`) (1.0); after that `temperature_late` (0) plays the most
  visited move.
- *Dirichlet noise*, on full searches only: the root priors become (1 - ε) P + ε Dir(α), with
  α = `dirichlet_alpha` (0.3) and ε = `dirichlet_epsilon` (0.25).
- *Random opening plies*: each game starts with `random_opening_plies` (8) uniformly random legal
  moves. They are in the history planes but give no records and do not count towards
  `temperature_plies` or `max_plies`.

**The search** (`cpp/mcts/mcts_small.cpp`, one tree per game) selects children by PUCT. A child's
score is Q + U + 0.05 (1 - 2W) tanh(plies / 25): Q is its mean value W - L from the parent's side
(0 when unvisited), U = `cpuct` P sqrt(N_parent) / (1 + N_child) with `cpuct` 0.65, W is its
backed-up win probability for the parent's side and plies its backed-up moves-left estimate. The
last term is a moves-left tiebreak: when W > 0.5 it prefers shorter lines, otherwise longer ones.
A leaf's value comes from the engine's `value` output (a WDL triple, scored W - L) and its
plies-remaining estimate from the `mlh_q` output (times 25); each ancestor accumulates the leaf's
estimate plus its distance to the leaf. After the search, the root's WDL is recomputed as the
visit-weighted average of its subtree. Checkmate, stalemate, insufficient material, the fifty-move
rule and a position occurring for the third time are exact results, both inside the tree and in
the game (chess-library's `isGameOver`: draws are automatic, not claimed). A game that reaches
`max_plies` (600) searched plies is a draw.

**Virtual-loss batching.** Each search descends `virtual_loss_width` (8) times before waiting for
the network. An in-flight descent adds a virtual visit to every node on its path, counted as a
loss for the side that chose it, so the next descent tends to pick another leaf; the 8 leaves are
then evaluated together. Width 1 is a strictly sequential search. All threads share one
`SmallBatchRunner` (`cpp/mcts/single_inference.cpp`), which dispatches a batch when `threads`
leaves are queued or after 100 µs, to four inference workers replaying CUDA graphs. Leaves travel
as packed bitboards (about 776 bytes) that a CUDA kernel (`cpp/mcts/expand_kernel.cu`) expands
into the 111 input planes on the GPU. `hash_size` > 0 adds a transposition table (in MB).

**Resignation.** The side to move resigns when its root expected score (W + D/2) has been below
`resign_threshold` (0.05) for `resign_consecutive` (3) of its own moves in a row. A fraction
`resign_playthrough` (0.1) of games never resign; in those, a would-be resignation counts as wrong
if that side did not lose (totals in `metrics/selfplay.jsonl`, used by `resign_target_fp`).

**Records.** A full search adds a pending record. At the game's end each record gets the result
from its side to move (one-hot WDL), the plies remaining (game length minus its ply) and the
game's flags, and the game's records are written together. Unfinished games are lost on shutdown.

## The buffer

`buffer.bin` holds fixed 1,280-byte records (`RECORD_DTYPE` in `src/train/dataset.py`, written by
`cpp/selfplay.cpp`, little-endian):

| Offset | Bytes | Field | Contents |
|---:|---:|---|---|
| 0 | 100 | `fen` | FEN, null-padded (the trainer reads the side to move from it) |
| 100 | 16 | `played` | the move played, UCI, null-padded |
| 116 | 12 | `value_outcome` | 3 x f32: the game result as one-hot WDL, from the side to move |
| 128 | 12 | `value_mcts` | 3 x f32: the root WDL after the search |
| 140 | 4 | `plies_remaining` | f32: plies from this position to the end of the game |
| 144 | 4 | `backed_plies` | f32: the search's backed-up plies remaining at the root, -1 if none |
| 148 | 7 | `past_rep` | 7 x u8: repetition flags of positions t-1 to t-7 |
| 155 | 1 | `cur_rep` | u8: repetition flag of the current position |
| 156 | 4 | `_pad` | padding |
| 160 | 672 | `past_bb` | 7 x 12 u64: bitboards of t-1 to t-7 (white PNBRQK, black PNBRQK), newest first; zero = no history |
| 832 | 96 | `cur_bb` | 12 u64: bitboards of the current position |
| 928 | 1 | `version` | 2 |
| 929 | 1 | `flags` | 1 full search, 2 move sampled (root temperature > 0), 4 resigned game, 8 playthrough game, 16 curriculum start, 32 drawn at `max_plies` |
| 930 | 1 | `castling` | WK 1, WQ 2, BK 4, BQ 8 |
| 931 | 1 | `ep` | en passant square 0 to 63 (a1 = 0), 255 = none |
| 932 | 8 | `halfmove`, `fullmove`, `ply`, `sims` | 4 x u16: halfmove clock, fullmove number, ply in the game (searched plies), simulations of this search |
| 940 | 2 | `n_policy` | u16 number of visit entries K |
| 942 | 2 | `played_idx` | u16 vocabulary index of the played move, 0xFFFF = none |
| 944 | 336 | `policy` | 84 x (u16 vocabulary index, u16 visit share x 65535), most visited first; the first K are used |

Records are appended until the file holds `max_buffer_size` (2,000,000: 2.56 GB). After that,
with `fifo_buffer: true` the file is a ring (write n goes to slot n mod capacity, replacing the
oldest record); with `false` each write replaces a random slot. After every step self-play
rewrites, by atomic rename, `buffer.bin.writes` (records ever written) and
`buffer.bin.stats.json`. A full buffer stops growing, so the trainer takes the insertion count and
the ring's head from `.writes`. `scripts/check_buffer.py` checks records against python-chess.

## The trainer

`src/train/__main__.py` builds a `Trainer` (`src/train/trainer.py`) from the config. It waits
until the buffer file holds `min_samples_to_start` (50,000) records, then loops.

**Pacing** is a token bucket. With P the records written since this process started training
(from `.writes`), the trainer steps only while its trained samples stay below `target_k` (5) times
P, and sleeps 0.2 s otherwise: on average it draws `target_k` samples per record written.

**Sampling.** A background thread prepares batches of `batch_size` (2,048) in pinned memory. Rows
are drawn uniformly from a window over the newest records that grows with N, the records ever
written (KataGo's rule): W = c (1 + β ((N/c)^α - 1) / α) for N > c and W = N below, capped by the
buffer, with c = `window_c` (250,000), α 0.75, β 0.4 (751k records at N = 2M, the whole buffer
from about 8.5M). The window needs the FIFO ring; otherwise the whole buffer is sampled. Torn
rows (wrong version byte) are drawn again. With
probability `mirror_prob` (0.5) a sample is colour-flipped (ranks mirrored;
colours, castling rights and side to move swapped; move indices remapped). The planes are decoded
from the bitboards on the GPU (`decode_planes`).

**Losses**, summed with the `loss_weights`:

| Head | Target | Loss | Weight |
|---|---|---|---:|
| policy | the record's visit shares, renormalised (`policy_target: visits`); or the move played (`played`) | cross-entropy | 1.0 |
| value | the game result, one-hot WDL | cross-entropy | 1.0 |
| value_q | the root WDL after the search | cross-entropy | 1.0 |
| mlh | plies remaining / 25, capped at 4 | squared error | 0.05 |
| mlh_q | the search's backed-up plies at the root / 25, capped at 4 | squared error | 0.05 |

Both policy losses are logged whichever is trained; rows without a target are masked out. Each
quantity has an *outcome* head (value, mlh: trained on what happened in the game) and a *search*
head (value_q, mlh_q: trained on the search's own estimate). `search_heads` (`outcome`) picks the
pair the engine gives the search; the other pair is still trained.

**Optimisation.** AdamW at `learning_rate` (1e-3) after a 1,000-step linear warmup, with weight
decay (0.05) on parameters of two or more dimensions only. The forward pass runs under
`torch.autocast` in float16 with a gradient scaler, the losses in FP32; the gradient norm is
clipped to 1.0 after unscaling.

**Logging and checkpoints.** Every `log_every` (5) steps a JSON row (losses, accuracies,
entropies, target statistics, pacing, window size, learning rate) goes to `metrics/train.jsonl`
and, if `wandb.enabled`, to W&B. Every `checkpoint.every_steps` (500) the trainer writes
`step_NNNNNNNN.pt` (weights, `model_spec`, optimizer, scaler, step, samples seen, last growth step
and the full config) through a temporary file and a rename, starts the engine build, and deletes
all but the newest `keep_last` checkpoints (the recipe keeps them all).

## Engine build and swap

After each checkpoint the trainer runs `scripts/build_trt.sh <ckpt> <step> <engine_dir>
<iteration_file>`, which:

1. exports ONNX with `python -m src.export.to_onnx`: the network is rebuilt from the checkpoint's
   `model_spec` and wrapped in `ExportWrapper`, whose four outputs are `policy` (logits), `value`
   (softmax WDL), `mlh` and `mlh_q`; the search reads `value` and `mlh_q`. With `search_heads`
   from the checkpoint's saved config, `outcome` routes the value head to `value` and the mlh head
   to `mlh_q`, and `search` routes value_q and mlh_q;
2. builds `step_NNNNNNNN.trt` with `trtexec --fp16` for batches of 1 to 512 (optimised for 256);
3. appends `<step>,<engine path>` to `.iteration`, a CSV with the header `iteration,engine_path`
   whose last row is the current engine.

With `async_engine_build: true` the build runs in the background (`engine_builds.log`); if the
previous build is still running, the new checkpoint gets no engine (a `build_skipped` event).
Self-play checks the modification time of `.iteration` every `selfplay.poll_seconds` (300). When
the last row names a new engine, `SelfPlayWorker.set_engine` replaces the inference runner between
two steps: games in progress continue with the new network and nothing restarts.

## Resume

`python -m src.run --resume <run_dir>` passes the newest `checkpoints/step_*.pt` as `resume_path`
and reuses `<run_dir>/config.yaml` (or `--config`; the paths are rewritten either way). If
`.iteration` names no engine (a run started from another run's checkpoint), it first builds one
from that checkpoint. The trainer rebuilds the network from the checkpoint's `model_spec` (a grown
network comes back grown) and loads the weights, the optimizer (re-initialised if incompatible),
the scaler, the step, the sample count and the last growth step; pacing counts from the restart.
Self-play reopens the buffer, cuts a partial trailing record, takes the write counter from
`.writes` (so the ring's head is right) and loads the last engine in `.iteration`.

## Models

`src/models/factory.py` builds a network from the `model_spec` saved in every checkpoint (the
architecture and its sizes; none means the 4 x 64 ResNet). Both networks take the same input, have
the same five heads, and return the policy logits, or all five outputs with `return_all=True`.

**Input** (`src/models/small/fen_encoder.py`; `decode_planes` and the C++ encoder match it):
8 x 8 x 111 planes in absolute orientation (not flipped for the side to move). Eight history slots
(the position and the 7 before it) of 12 piece planes and a repetition flag, then side to move,
en passant square, the four castling rights and the halfmove clock / 100 (capped at 1). The
**policy vocabulary** (`src/models/small/vocab.py`, `cpp/policy_index.txt`) has 1,929 entries:
1,924 UCI moves (castling as the king's move, knight promotions sharing the plain move's entry)
and 5 special tokens that are never played.

**Encoder** (`src/models/encoder.py`, the recipe; 812,557 parameters at d_model 128, 4 layers,
8 heads, d_ff 256):

- One token per square: its 111 plane values, linearly embedded, plus a learned position
  embedding. This is x0. Every block first adds `x0_gain · x0` to the stream (x0 re-injection;
  the gain starts at 0), then runs pre-norm attention and a SwiGLU feed-forward,
  `down(silu(gate) · up)`, each with a residual connection.
- Attention uses QK-norm (RMSNorm on each head's queries and keys) and adds a learned square-pair
  bias (heads x 64 x 64) shared by all layers. It is plain matmul and softmax, which the ONNX
  export and TensorRT handle.
- RMSNorm divides each row by its largest absolute value (detached) before squaring. RMSNorm is
  scale-invariant, so the function is unchanged up to eps, but x² stays inside FP16 range (it
  overflows past |x| = 256) when TensorRT runs the norm in FP16.
- Policy head: logit(from, to, promotion) = (q_from + p_promotion) · k_to / sqrt(d_model), with q
  and k linear projections of the final tokens and p one of four learned vectors (none, queen,
  rook, bishop). The special tokens share one learned logit, initialised at -10.
- value, value_q, mlh, mlh_q: the mean of the 64 tokens, then a two-layer MLP each (3 WDL logits,
  or one scalar in units of 25 plies).

**ResNet** (`src/models/small/small_model.py`, `model.arch: resnet`): a 3 x 3 convolution stem and
`blocks` residual blocks of two 3 x 3 convolutions with batch norm, `channels` wide (4 x 64:
1,285,921 parameters). Each head is a 1 x 1 convolution and two fully connected layers.

**Growth** (`src/models/growth.py`, off in the recipe): `growth.schedule` enlarges the trunk
during training without changing the output at that moment. New encoder layers start with zero
attention output, feed-forward down projection and x0 gain (identity blocks); new ResNet blocks
start with a zeroed final batch norm, and new channels with zero weights into the old channels and
heads. The trainer then rebuilds the optimizer and re-warms the learning rate.

## Evaluation tools

- **Rating ladder** (`src/rating/ladder.py`, `scripts/ladder_daemon.py`). Every 30 minutes the
  daemon rates the run's newest engine into `<run>/ladder/ladder.jsonl`: 1,000 Lichess puzzles
  (every model move of the line must match), the first-move preferences, and 48 games (24
  balanced 6-ply openings chosen by Stockfish, both colours) against each of the two anchors
  nearest its previous rating, plus up to 3 more while it scores 85% or more against the strongest
  played (or 15% or less against the weakest). The anchors (random mover, greedy capturer,
  node-limited Stockfish at several skill levels, some mixing in random moves) were rated once by a
  round robin (`--calibrate`, `src/rating/anchors_sf.json`, random = 0); the model's rating is the
  maximum-likelihood fit against those fixed ratings, with two virtual draws at its previous
  rating as a prior. The network searches 100 simulations per move (cpuct 0.65, no noise, most
  visited move, virtual-loss width 1); games use the self-play draw rules and end drawn at 300
  plies. `scripts/rerate.py` rates saved checkpoints the same way.
- **The first post's Stockfish test**: `scripts/stockfish_match.py` plays an engine (800
  simulations per move) against Stockfish limited by `UCI_LimitStrength` and `UCI_Elo` at 100 ms
  per move, from the initial position with alternating colours. `scripts/sf_benchmark.py` runs
  several engines and levels in parallel; `scripts/cpu_stockfish_match.py` uses the CPU search.
- **Engine parity**: `scripts/trt_parity.py` compares each checkpoint's FP16 engine with PyTorch
  FP32 on puzzle positions (top-1 agreement, policy KL, value error).
- **Dashboard and analysis board**: `scripts/dashboard.py` (Streamlit, read-only) reads the metrics
  and ladder files of the runs under `artifacts/runs/` or `$CHESSRL_RUNS_ROOT`. Its analysis page
  embeds `scripts/analysis_server.py`, which runs a checkpoint's policy, value and PUCT search on
  the CPU (`src/analysis/engine.py`, following the self-play search's conventions).

## Where things live

| Concern | Code |
|---|---|
| Launcher | `src/run.py` |
| Config schema and loading; the recipe | `src/train/config.py`; `src/train/config.yaml` |
| Trainer; buffer reader, record layout, plane decoding | `src/train/trainer.py`, `src/train/__main__.py`; `src/train/dataset.py` |
| Self-play process | `src/self_play/__main__.py`, `engine_watch.py` (`.iteration`), `calibration.py` (resign threshold) |
| Self-play worker and record writer | `cpp/selfplay.cpp`, `cpp/selfplay.hpp` |
| MCTS | `cpp/mcts/mcts_small.cpp`, `cpp/mcts/common.cpp`, `cpp/mcts/common.hpp`, `cpp/mcts/cache.hpp` (transposition table) |
| Batched inference, CUDA graphs; input layout, packed positions, CPU encoders | `cpp/mcts/single_inference.cpp`, `cpp/mcts/expand_kernel.cu`; `cpp/mcts/small_layout.hpp`, `cpp/mcts/packed_position.hpp`, `cpp/mcts/small_encoding.cpp` |
| TensorRT helpers; C++ policy vocabulary | `cpp/inference.cpp`, `cpp/include/`; `cpp/policy_vocab.cpp`, `cpp/policy_index.txt` |
| Python bindings (`_inference_cpp`), build, chess rules | `cpp/bindings.cpp`, `cpp/setup.py`, `cpp/pyproject.toml`, `cpp/chess-library` (submodule) |
| Networks, input planes, policy vocabulary | `src/models/encoder.py`, `src/models/small/small_model.py`, `src/models/factory.py`, `src/models/growth.py`; `src/models/small/fen_encoder.py`, `src/models/small/vocab.py` |
| Export and engine build | `src/export/to_onnx.py`, `scripts/build_trt.sh` |
| Rating ladder | `src/rating/ladder.py`, `src/rating/anchors.json`, `src/rating/anchors_sf.json`, `scripts/ladder_daemon.py`, `scripts/rerate.py`, `scripts/calibrate_finish.py`; data in `data/openings/`, `data/puzzles/` (built by `scripts/make_sf_openings.py`, `scripts/make_puzzle_set.py`) |
| Stockfish test | `scripts/stockfish_match.py`, `scripts/sf_benchmark.py`, `scripts/cpu_stockfish_match.py` |
| CPU search, dashboard, analysis board | `src/analysis/engine.py`, `scripts/puzzle_scan.py`; `scripts/dashboard.py`, `scripts/pages/1_Analysis.py`, `scripts/analysis_server.py`, `scripts/analysis/` |
| Diagnostics and benchmarks | `scripts/trt_parity.py`, `scripts/check_buffer.py`, `scripts/bench_selfplay.py`, `scripts/perfreport.py`, `scripts/showcase_games.py`; CPU search benchmark (fake network, no GPU): `cpp/bench/`, `scripts/build_cpu_bench.sh`, `scripts/bench_cpu_search.sh`, `scripts/make_bench_positions.py`, `data/bench/positions.txt`; results in `docs/PERF.md` |
| Rented GPUs | `scripts/vast/` |
| Tests | `tests/` (Python); `cpp/mcts/tests.cpp`, built by `scripts/build_cpp_tests.sh` (C++) |
| Lab log and blog posts | `docs/NEXT_ROUND.md`, `docs/blog/` |
