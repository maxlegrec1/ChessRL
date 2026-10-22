#!/bin/bash
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
ENGINE_DIR_ABS="$(realpath "$ENGINE_DIR")"
MODEL_NAME="$(basename "${CKPT_PATH%.*}")"
ONNX_PATH="${ENGINE_DIR_ABS}/${MODEL_NAME}.onnx"
TRT_PATH="${ENGINE_DIR_ABS}/${MODEL_NAME}.trt"

echo "[build_trt] Exporting ${CKPT_ABS} -> ${ONNX_PATH}"
uv run python -m src.export.to_onnx \
  --ckpt_path "$CKPT_ABS" \
  --onnx_path "$ONNX_PATH"

echo "[build_trt] Building TensorRT engine ${TRT_PATH}"
# Max batch 512: each execution context is sized for the profile's max shape,
# and an in-place engine swap holds the old and new contexts at once. At 1024 an
# 8-layer encoder's context needed 723 MB and the swap went out of memory.
# Self-play batches are capped at its thread count (CUDA-graph buckets <= 256)
# and the ladder runs at 64; other tools must set CHESSRL_MCTS_BATCH_SIZE <= 512.
trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$TRT_PATH" \
  --fp16 \
  --minShapes=input:1x8x8x111 \
  --optShapes=input:256x8x8x111 \
  --maxShapes=input:512x8x8x111 \
  --shapes=input:256x8x8x111 \

TRT_ABS="$(realpath "$TRT_PATH")"
ITER_FILE_PATH="$ITERATION_FILE"
mkdir -p "$(dirname "$ITER_FILE_PATH")"
if [ ! -f "$ITER_FILE_PATH" ]; then
  echo "iteration,engine_path" > "$ITER_FILE_PATH"
fi
echo "${STEP},${TRT_ABS}" >> "$ITER_FILE_PATH"
echo "[build_trt] Updated ${ITER_FILE_PATH} with ${TRT_ABS}"