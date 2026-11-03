# ChessRL architecture

How the code works: the two processes and the files they share, self-play and the search, the
buffer, the trainer, the network and its export, and the tools that measure a run. Numbers in
parentheses are the recipe's values (`src/train/config.yaml`); every setting is described in
[`CONFIG.md`](CONFIG.md).

## Processes and files

`uv run python -m src.run` (`src/run.py`) creates `artifacts/runs/<YYYYmmdd_HHMMSS>_<4 hex>[_<name>]/`,
exports a randomly initialised network so self-play has one from the start, writes the config to
`<run>/config.yaml` with every path inside the run, and starts `python -m src.self_play` and
`python -m src.train` on it (output in `selfplay.log` and `training.log`). When one process exits,
the launcher stops the other. Under NVIDIA MPS both processes run their kernels on the GPU at the
same time. They share nothing but files:

```
artifacts/runs/<id>/
├── config.yaml                       the resolved config; both processes read it
├── buffer/buffer.bin                 fixed-size records: self-play writes, the trainer memory-maps
├── buffer/buffer.bin.writes          records ever written (pacing, and the ring's head)
├── checkpoints/step_NNNNNNNN.pt      trainer checkpoints
├── engines/step_NNNNNNNN.cnet        networks exported from them
├── engines/.iteration                CSV of published networks: self-play plays the last row
├── metrics/train.jsonl, selfplay.jsonl  (and diversity.jsonl from scripts/opening_diversity.py)
├── ladder/ladder.jsonl               ratings from scripts/ladder_daemon.py
└── selfplay.log, training.log, engine_builds.log
```

Either process can be stopped and started again on its own. `python -m src.run --resume <run_dir>`
resumes from the newest checkpoint (weights, optimizer, scaler, step) with the run's config;
self-play reopens the buffer and takes the write counter from `.writes`. A run started from another
run's checkpoint should start with a copy of that run's `buffer.bin*`: an empty buffer gives the
trainer a tiny sampling window and the value head overfits it for about an hour.

## Self-play

`src/self_play/__main__.py` copies the `selfplay` settings onto the C++ worker
(`cpp/selfplay.cpp`). One `worker.step()` plays one move in each of the `boards` (1,024) games,
running their searches on `threads` (128) CPU threads, and writes the records of finished games.
Between steps the Python loop looks for a new network and logs to `metrics/selfplay.jsonl`.

- **Playout-cap randomisation**: a move gets a full, recorded search of `simulations` (400) with
  probability `pcr_full_prob` (0.25), else an unrecorded `pcr_fast_simulations` (50) search that only
  moves the game forward.
- **Tree reuse**: the next search continues the subtree of the move played, so only the missing
  simulations run.
- **Move choice**: for the first `temperature_plies` (40) searched plies the move is sampled with
  probability proportional to visits^(1/T), with T = `temperature_at_root` (1.0) on full searches
  and `fast_temperature_at_root` (0.7) on fast ones, whose 50-simulation visit counts are flat.
  After that the most visited move is played. Games start from the initial position
  (`random_opening_plies` 0).
- **Dirichlet noise** on full searches only: root priors (1 - ε) P + ε Dir(α), α 0.3, ε 0.25.
- **Resignation**: the side to move resigns when its root expected score (W + D/2) stays below 0.05
  for 3 of its own moves. 10% of games never resign; in those, a would-be resignation is checked
  against the result (the false-resignation rate in `metrics/selfplay.jsonl`).
- **Records**: every full search adds a record; at the game's end each one gets the result from its
  side to move and the plies remaining, and the game's records are written together.

## The search

`cpp/mcts/mcts_small.cpp` keeps one tree per game and selects children by PUCT: Q + U plus a small
moves-left term, with U = `cpuct` (0.65) P sqrt(N_parent) / (1 + N_child) and priors
softmax(logits / `temperature`) (1.3). The moves-left term prefers shorter lines when winning and
longer ones when losing. A leaf's value is the network's WDL scored W - L; mate, stalemate,
insufficient material, the fifty-move rule and threefold repetition are exact results, inside the
tree and in the game. A game that reaches 600 searched plies is a draw.

Each search descends `virtual_loss_width` (8) times before waiting for the network: an in-flight
descent counts as a loss on its path, so the next one picks another leaf. All threads share one
batch runner (`cpp/mcts/single_inference.cpp`) that sends batches to the GPU on several CUDA
streams. Leaves travel as packed bitboards; the network kernel expands them into input planes.

## The buffer

`buffer.bin` is a ring of `max_buffer_size` (6,000,000) records of 1,280 bytes (7.7 GB): write n
goes to slot n mod capacity, replacing the oldest record. The layout is `RECORD_DTYPE` in
`src/train/dataset.py` (written by `cpp/selfplay.cpp`): the position (FEN and bitboards of the
position and its 7 predecessors), castling and en passant, the visit counts (up to 84 moves), the
root WDL after the search, the game result, the plies remaining and flags. Once full, the file
stops growing, so the insertion count lives in `buffer.bin.writes`. `scripts/check_buffer.py`
checks records against python-chess.

## The trainer

`src/train/trainer.py` waits until the buffer holds `min_samples_to_start` (50,000) records.

- **Pacing** is a token bucket on `.writes`: the trainer draws at most `target_k` (5) samples per
  record written since it started, and sleeps otherwise.
- **Sampling**: batches of 2,048 rows drawn uniformly from a window over the newest records that
  grows with N, the records ever written (KataGo's rule: W = c (1 + β ((N/c)^α - 1) / α), c 250,000,
  α 0.75, β 0.4), capped by the buffer. A background thread prepares them in pinned memory and the
  planes are decoded on the GPU.
- **Losses**: policy against the visit shares (cross-entropy, weight 1); `value` against the game
  result and `value_q` against the search's root WDL (cross-entropy, 1 each); `mlh` and `mlh_q`
  against the plies remaining and the search's backed-up estimate (squared error, 0.05 each). The
  search reads the outcome-trained heads (`search_heads: outcome`); the others are trained as
  auxiliary targets and diagnostics.
- **Optimisation**: AdamW at 1e-3 after a 1,000-step warmup, flat to step 70,000, then a cosine down
  to 1e-4 at step 280,000; weight decay 0.05 on matrices only; FP16 autocast with a gradient scaler;
  gradient norm clipped to 1. `optimizer: muon` puts Muon on the encoder's matrices instead.
- **Speed**: the step is compiled (`torch.compile`) and runs Triton kernels that fuse each encoder
  block (`src/models/fused_*.py`), with the residual stream in FP16.
- **Checkpoints and logs**: a JSON row to `metrics/train.jsonl` every 5 steps, and a checkpoint every
  500 steps (weights, `model_spec`, optimizer, scaler, step and the full config).

## Export and swap

After each checkpoint the trainer runs `scripts/build_net.sh` in the background: it exports the
checkpoint to `step_NNNNNNNN.cnet` with `python -m src.export.to_cnet` (about 0.1 s) and appends
`<step>,<path>` to `.iteration`. Self-play checks `.iteration` every `poll_seconds` (300) and swaps
the new network in between two steps; games in progress continue with it.

## The network

**Canonical positions.** The network always plays white: a position with black to move is mirrored
(ranks reversed, colours, castling rights and en passant swapped, history included), and black's
moves are looked up the same way. The mirror happens once, where a position enters the system
(the C++ packer, `fen_to_tensor` in Python, and the self-play records, which are stored canonical).
Nothing downstream depends on the colour.

**Input**: 8 x 8 x 111 planes (`src/models/small/fen_encoder.py`, matched by the C++ encoder): the
position and its 7 predecessors (12 piece planes and a repetition flag each), side to move, en
passant, castling rights and the halfmove clock. **Policy vocabulary**: white's 1,858 moves
(`src/models/small/vocab.py`, `cpp/policy_index.txt`).

**Encoder** (`src/models/encoder.py`; 812,557 parameters at width 128, 4 layers, 8 heads,
feed-forward 256):

- One token per square: its 111 plane values, embedded, plus a learned position embedding (x0).
  Each block adds a learned multiple of x0 to the stream, then runs pre-norm attention and a SwiGLU
  feed-forward, each with a residual connection.
- Attention has QK-norm and a learned square-pair bias shared by all layers. RMSNorm divides each
  row by its largest value before squaring, so the squares cannot overflow in FP16.
- Policy: logit(from, to, promotion) = (q_from + p_promotion) · k_to / sqrt(128), one logit per
  vocabulary entry.
- `value`, `value_q` (WDL) and `mlh`, `mlh_q` (plies left / 25): a two-layer MLP each on the mean of
  the 64 tokens.

**Growth** (`src/models/growth.py`, off in the recipe) adds encoder layers during training without
changing the network's output at that moment.

**The fused network.** Self-play, the ladder and the GPU tools run the network in one CUDA kernel
(`cpp/net/encoder_kernel.cu`), one launch per batch and one thread block per position. The `.cnet`
file holds the weights in FP16, pre-packed in the order the tensor-core instructions read them, so
the loader (`cpp/net/encoder_net.cpp`) only copies it to the GPU. Activations stay in registers and
shared memory; norms and accumulation are FP32. The kernel is written for width 128, 8 heads,
feed-forward 256 and up to 16 layers; the export refuses other sizes. `scripts/net_parity.py`
compares it with PyTorch on real positions.

## Measuring a run

- **Rating ladder** (`src/rating/ladder.py`, `scripts/ladder_daemon.py`, every 30 minutes by
  default): 48 games against each of the two reference opponents nearest the network's previous
  rating, from 24 balanced openings chosen by Stockfish, both colours, at 100 simulations per move
  without noise. The opponents (a random mover, a greedy capturer, reduced Stockfish levels) were
  rated once among themselves with random play at 0; the network's rating is the maximum-likelihood
  fit against them, about ±45 Elo. Each rating also solves 1,000 Lichess puzzles and records the
  network's first-move preferences.
- **Opening diversity** (`scripts/opening_diversity.py`): for each ply, how many distinct positions
  the newest 400,000 records hold (2^entropy) and the share of the most common one.
- **Dashboard** (`scripts/dashboard.py`, page in `scripts/dashboard_ui/`): a small read-only HTTP
  server that reads every run's metrics and ladder files incrementally and draws them by hours,
  steps or positions.
- **Analysis board** (`scripts/analysis_server.py`): a checkpoint's policy, value and search on the
  CPU for any position (`src/analysis/engine.py`), and a viewer for self-play PGN samples.

## Where things live

| Concern | Code |
|---|---|
| Launcher; config and recipe | `src/run.py`; `src/train/config.py`, `src/train/config.yaml` |
| Trainer; buffer reader and record layout | `src/train/trainer.py`; `src/train/dataset.py` |
| Self-play process; worker | `src/self_play/`; `cpp/selfplay.cpp` |
| MCTS; batched inference; packed positions | `cpp/mcts/mcts_small.cpp`, `common.cpp`; `single_inference.cpp`; `packed_position.hpp`, `small_encoding.cpp` |
| Fused network | `cpp/net/` |
| Python bindings, build | `cpp/bindings.cpp`, `cpp/setup.py`; chess rules in `cpp/chess-library` (submodule) |
| Network, input, vocabulary | `src/models/encoder.py`, `fused_*.py`, `factory.py`, `growth.py`; `src/models/small/` |
| Export | `src/export/to_cnet.py`, `heads.py`, `scripts/build_net.sh` |
| Ladder; data | `src/rating/`, `scripts/ladder_daemon.py`; `data/openings/`, `data/puzzles/` |
| Rented GPUs | `scripts/vast/` |
| Tests | `tests/` (Python), `cpp/mcts/tests.cpp` (C++, `scripts/build_cpp_tests.sh`) |
