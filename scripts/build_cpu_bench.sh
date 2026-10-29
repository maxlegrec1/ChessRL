#!/usr/bin/env bash
# Build cpp/build/bench/bench_search: the self-play search (production code) with a fake
# network in place of TensorRT, so it builds and runs on any machine with g++, no GPU.
# Same compile flags as cpp/setup.py, plus -g for profiler symbols (code generation is
# unchanged). Run it from the repo root (the policy vocabulary is read from cpp/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP="$ROOT/cpp"
OUT_DIR="$CPP/build/bench"
CXX="${CXX:-g++}"

mkdir -p "$OUT_DIR"

SOURCES=(
    "$CPP/bench/bench_search.cpp"
    "$CPP/bench/fake_network.cpp"
    "$CPP/mcts/mcts_small.cpp"
    "$CPP/mcts/common.cpp"
    "$CPP/mcts/small_encoding.cpp"
    "$CPP/policy_vocab.cpp"
)

"$CXX" -O3 -march=native -DNDEBUG -std=c++17 -DCHESSRL_MCTS_NO_STANDALONE -g -pthread \
    "-I$CPP" "-I$CPP/include" "-I$CPP/chess-library/include" \
    "${SOURCES[@]}" -o "$OUT_DIR/bench_search"

echo "built $OUT_DIR/bench_search"
