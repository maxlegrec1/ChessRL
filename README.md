# ChessRL

AlphaZero for chess, trained from scratch on a single RTX 4090.

ChessRL learns chess only by playing against itself: no human games, no opening book, no endgame
tables. C++ workers play 1,024 games at once with Monte Carlo tree search (MCTS) and run the network
in one fused CUDA kernel; a PyTorch trainer learns from the positions they save and publishes new
networks back into the running games every 500 steps.

## Where it stands

On a single RTX 4090 with a Ryzen 9 7950X, ratings on the project's ladder (random play = 0, about
±45 per rating) and the share of 1,000 Lichess puzzles solved:

| Recipe | Around 4 h | Later | Puzzles solved |
|---|---|---|---|
| Previous recipe (constant learning rate) | ~2,300 | levels off at 2,300-2,400 | 0.73 (3.3 h) |
| **Current recipe** (`src/train/config.yaml`) | ~2,360 | **~2,520 at 8 h** | **0.80 (8 h)** |

How we got here, and what did not work, is in [`docs/RESULTS.md`](docs/RESULTS.md). The two blog
posts tell the longer story:

- [AlphaZero from scratch: training a chess AI on a single consumer GPU](https://mvdbk.fr/blog/chessrl-alphazero-single-gpu/)
  (the first version, commit `128893c`)
- [AlphaZero from scratch, continued: 6× faster on a single RTX 4090](https://mvdbk.fr/blog/alphazero-from-scratch-rtx-4090/)

## Quickstart

You need an NVIDIA GPU with compute capability 8.0 or higher, the CUDA Toolkit 12.x with `nvcc`, a
C++17 compiler, [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 itself), and Stockfish 16
for the rating ladder (`apt install stockfish`; training runs without it).

```bash
git clone --recurse-submodules https://github.com/maxlegrec1/ChessRL.git && cd ChessRL
bash scripts/check_install.sh            # checks uv, the GPU, nvcc, Stockfish; hints for anything missing
export CHESSRL_CUDA_ARCH=sm_89           # sm_89 RTX 4090, sm_90 H100/H200, sm_86 RTX 3090, sm_80 A100
uv sync                                  # builds the C++ extension, installs the Python dependencies

uv run python -m src.run --name my-run                 # a run with the current recipe
uv run python -m src.run --resume artifacts/runs/<id>  # resume from its latest checkpoint
uv run python scripts/ladder_daemon.py --run-dir artifacts/runs/<id>   # rate it every 30 min
uv run python scripts/dashboard.py --port 8501         # http://localhost:8501
uv run python scripts/analysis_server.py --port 8510   # play through positions with a checkpoint
```

A run lives in `artifacts/runs/<id>/`: its config, buffer, checkpoints, exported networks, metrics
(`metrics/*.jsonl`), ladder ratings and logs. The dashboard reads every run there.

If `uv sync` cannot find `nvcc`, point `CUDA_ROOT` at your CUDA install (for example
`/usr/local/cuda-12.8`). If self-play is slow, the extension was built for another GPU: set
`CHESSRL_CUDA_ARCH` and run `uv sync` again.

## How it works

- `src/run.py` starts two processes that share nothing but files: self-play and the trainer.
- **Self-play** (`src/self_play/`, C++ in `cpp/`) writes the positions of finished games to a ring
  buffer file, with the search's visit counts and the game result.
- **The trainer** (`src/train/`) samples batches from the newest part of that buffer and trains the
  network to predict the visit counts, the result and the number of moves left.
- Every 500 steps it saves a checkpoint and exports it to `.cnet`, the fused kernel's format;
  self-play loads the new network between moves, without restarting.
- The network sees every position from the side to move, as white, and scores moves from one
  1,858-move vocabulary.
- Strength is measured by a rating ladder against fixed opponents (random players and reduced
  Stockfish levels) from balanced openings, plus 1,000 puzzles.

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has the details.

## The current recipe

- **Network**: a transformer with one token per square, 0.81M parameters.
- **Targets**: the policy learns the search's visit counts; the search reads the value trained on
  game results.
- **Search**: 25% of moves get a full 400-simulation search (recorded), 75% a 50-simulation one that
  only moves the game forward. Dirichlet noise on full searches.
- **Openings**: every game starts from the initial position; the first 40 searched plies are
  sampled from the visit counts (full searches at temperature 1, fast ones at 0.7), then the most
  visited move is played.
- **Buffer**: a 6M-record ring (7.7 GB), sampled through a window that grows with the run.
- **Optimiser**: AdamW at 1e-3, flat to step 70,000 (~2 h), then a cosine down to 1e-4 at step
  280,000 (~8 h).

Every setting is described in [`docs/CONFIG.md`](docs/CONFIG.md).

## Repository map

| Path | What is there |
|---|---|
| `src/run.py` | The launcher |
| `src/self_play/` | The self-play process (drives the C++ worker) |
| `src/train/` | The trainer, the buffer reader, the config (`config.yaml` is the recipe) |
| `src/models/` | The transformer, its fused training kernels, growth |
| `src/export/` | Checkpoint to `.cnet` |
| `src/rating/`, `src/analysis/` | The ladder; the CPU search behind the analysis board |
| `cpp/` | MCTS, self-play, position encoding, the fused CUDA network (`cpp/net/`) |
| `scripts/` | Tools (below) |
| `tests/` | Python tests (C++ tests in `cpp/mcts/tests.cpp`) |
| `data/` | The ladder's openings and puzzles |

Scripts you will use: `check_install.sh`, `build_net.sh` (export a checkpoint by hand),
`ladder_daemon.py`, `dashboard.py`, `analysis_server.py`, `net_parity.py` (fused network against
PyTorch), `check_buffer.py` (buffer records against python-chess), `bench_selfplay.py` (self-play
speed), `opening_diversity.py` (how varied the self-play openings are), and `vast/` (runs on rented
GPUs).

## Contributing

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest        # CPU tests; GPU tests skip
uv run pytest                                # everything, on a machine with CUDA
bash scripts/build_cpp_tests.sh && cpp/build/tests/cpp_tests
uv run ruff check . && uv run ruff format --check .
```

Changes that touch self-play or the network should come with a speed or strength measurement: a
benchmark (`scripts/bench_selfplay.py`) or ladder ratings from a run against the current recipe.

## License

MIT.
