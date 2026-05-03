# ChessRL

AlphaZero-style chess training. C++ MCTS self-play feeds a PyTorch trainer that
publishes new TensorRT engines back to self-play in a continuous loop.

## Install

    uv sync

This builds the C++ inference extension (`_inference_cpp`) automatically as a
uv workspace member. Override the native paths via environment variables:

    TENSORRT_ROOT=/path/to/TensorRT CUDA_ROOT=/path/to/cuda uv sync

## Run

All commands assume the repo root as the working directory.

    # Start (or resume) the full training loop. This launches self-play and the
    # trainer in parallel; the trainer waits for the buffer to fill (see
    # data.min_samples_to_start in config.yaml) before its first step.
    uv run python -m src.run

    # Watch the per-process logs:
    tail -f artifacts/runs/selfplay.log
    tail -f artifacts/runs/training.log

    # Run only one side (useful for resume / debugging)
    uv run python -m src.run --no-train      # self-play only
    uv run python -m src.run --no-selfplay   # trainer only

    # Manually convert a checkpoint to a TensorRT engine
    scripts/build_trt.sh artifacts/checkpoints/step_00043900.pt 43900

    # Head-to-head evaluation between two checkpoints
    uv run python -m src.eval.brawls \
        --num_games 100 --model1 small --model2 small \
        --ckpt1 artifacts/checkpoints/<a>.pt \
        --ckpt2 artifacts/checkpoints/<b>.pt

    # Prune artifacts (keep only last N step checkpoints + matching engines)
    uv run python scripts/prune_artifacts.py            # dry-run
    uv run python scripts/prune_artifacts.py --apply

## Tests

    uv run pytest                            # full suite (skips gpu/trt if unavailable)
    uv run pytest -m "not gpu and not trt"   # CPU-only

## Layout

- `src/` — Python source
  - `models/{small,leela}/` — model architectures and inference wrappers
  - `train/` — training loop and dataset
  - `self_play/` — self-play orchestration (drives the C++ worker)
  - `eval/` — head-to-head and Stockfish evaluation
  - `export/` — checkpoint → ONNX export
  - `generate/` — opening-position generators
  - `mcts/` — Python MCTS bindings
- `cpp/` — pybind11 C++ inference extension (uv workspace member)
- `scripts/` — `build_trt.sh`, `prune_artifacts.py`
- `tests/` — pytest suite
- `artifacts/` — generated state (checkpoints, engines, buffers, runs, wandb); gitignored
- `archive/` — legacy scripts and historical model files
