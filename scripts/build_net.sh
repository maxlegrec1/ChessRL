#!/bin/bash
# Export a checkpoint to the fused network format (.cnet, src/export/to_cnet.py) and
# append it to the run's .iteration file, which self-play and the ladder watch:
#   build_net.sh CHECKPOINT_PATH STEP [ENGINE_DIR] [ITERATION_FILE]
# The trainer calls it after every checkpoint (checkpoint.build_script).
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 CHECKPOINT_PATH STEP [ENGINE_DIR] [ITERATION_FILE]"
  exit 1
fi

CKPT_PATH="$1"
STEP="$2"
ENGINE_DIR="${3:-artifacts/engines}"
ITERATION_FILE="${4:-artifacts/engines/.iteration}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p "$ENGINE_DIR"
CKPT_ABS="$(realpath "$CKPT_PATH")"
NET_PATH="$(realpath "$ENGINE_DIR")/$(basename "${CKPT_PATH%.*}").cnet"

uv run python -m src.export.to_cnet --ckpt_path "$CKPT_ABS" --out "$NET_PATH"

mkdir -p "$(dirname "$ITERATION_FILE")"
if [ ! -f "$ITERATION_FILE" ]; then
  echo "iteration,engine_path" > "$ITERATION_FILE"
fi
echo "${STEP},${NET_PATH}" >> "$ITERATION_FILE"
echo "[build_net] ${ITERATION_FILE} -> ${NET_PATH}"
