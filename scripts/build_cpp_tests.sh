#!/usr/bin/env bash
# Build the doctest-based C++ test binary at cpp/build/tests/cpp_tests.
# Mirrors the compile flags used by cpp/setup.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP="$ROOT/cpp"
OUT_DIR="$CPP/build/tests"
OUT_BIN="$OUT_DIR/cpp_tests"

TENSORRT_ROOT="${TENSORRT_ROOT:-/usr/local/TensorRT-10.14.1.48}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
CXX="${CXX:-g++}"

mkdir -p "$OUT_DIR"

SOURCES=(
    "$CPP/mcts/tests.cpp"
    "$CPP/mcts/common.cpp"
    "$CPP/inference.cpp"
)

INCLUDES=(
    "-I$CPP"
    "-I$CPP/include"
    "-I$CPP/chess-library/include"
    "-I$TENSORRT_ROOT/include"
    "-I$CUDA_ROOT/include"
)

LIB_DIRS=(
    "-L$TENSORRT_ROOT/lib"
    "-L$CUDA_ROOT/lib64"
)

LIBS=(-lnvinfer -lcudart -lpthread)

CXXFLAGS=(-std=c++17 -O2 -DCHESSRL_MCTS_NO_STANDALONE -Wall -Wno-unknown-pragmas)

echo "[build_cpp_tests] -> $OUT_BIN"
"$CXX" "${CXXFLAGS[@]}" "${INCLUDES[@]}" "${SOURCES[@]}" "${LIB_DIRS[@]}" "${LIBS[@]}" -o "$OUT_BIN"
echo "[build_cpp_tests] OK"
