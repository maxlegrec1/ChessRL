#!/usr/bin/env bash
# Self-play speed A/B on a rented GPU box (after scripts/vast/setup_box.sh), results in
# docs/PERF.md. Runs the GPU tests and the parity check, exports the network once, then
# alternates the arms ROUNDS times (default 2), each 60 s of warm-up and 240 s measured.
# An arm is NAME=CONFIG (a YAML that may `extends:` the recipe); ENV_<NAME>="K=V ..." adds
# environment variables to that arm. Each result line has the recorded positions per
# second and the CPU time per recorded position, with the host's load average: on a
# shared host a busy neighbour slows the search, so compare arms run close together.
#   cd /workspace/ChessRL && ARMS="recipe=src/train/config.yaml reuse=/workspace/perf/reuse.yaml" \
#     nohup bash scripts/perf/selfplay_ab.sh > /workspace/perf/ab.out 2>&1 &
# Needs a checkpoint at /workspace/perf/step_00092500.pt (CKPT=... to choose it).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /workspace/env.sh  # (it cds to the box's main checkout)
cd "$REPO"
P=/workspace/perf
CKPT="${CKPT:-$P/step_00092500.pt}"
ROUNDS="${ROUNDS:-2}"
ARMS="${ARMS:-recipe=src/train/config.yaml}"
mkdir -p "$P"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$P/ab.log"; }

log "commit $(git log --oneline -1); $(nvidia-smi --query-gpu=name --format=csv,noheader); $(lscpu | grep -m1 'Model name' | sed 's/  */ /g'); CPU quota $(awk '{printf "%.1f", $1 / $2}' /sys/fs/cgroup/cpu.max 2>/dev/null) cores"
(bash scripts/build_cpp_tests.sh && cpp/build/tests/cpp_tests) > "$P/cpp_tests.log" 2>&1
log "C++ tests: $(grep 'test cases' "$P/cpp_tests.log")"
uv run pytest -q tests/test_fused_network.py tests/test_cnet_export.py tests/test_inference_cpp.py \
  tests/test_encoder_parity.py tests/test_packed_parity.py > "$P/pytest_gpu.log" 2>&1
log "GPU tests: $(tail -1 "$P/pytest_gpu.log")"
uv run python scripts/net_parity.py "$CKPT" --n 256 > "$P/net_parity.log" 2>&1
log "parity: $(tail -1 "$P/net_parity.log")"
uv run python -m src.export.to_cnet --ckpt_path "$CKPT" --out "$P/net.cnet" > "$P/cnet.log" 2>&1

for round in $(seq 1 "$ROUNDS"); do
  for arm in $ARMS; do
    name="${arm%%=*}"
    cfg="${arm#*=}"
    envvar="ENV_$name"
    label="${name}_$round"
    env ${!envvar:-} uv run python scripts/bench_selfplay.py --engine "$P/net.cnet" --config "$cfg" \
      --warmup-sec 60 --duration-sec 240 --label "$label" --out "$P/ab_$label.json" > "$P/ab_$label.log" 2>&1
    log "$label: $(grep -o '[0-9.]* positions/sec.*' "$P/ab_$label.log" | head -1) |" \
      "$(grep -o 'inference: .*' "$P/ab_$label.log" | head -1) | $(grep -o 'cpu: .*' "$P/ab_$label.log" | head -1)"
  done
done
log "== done"
date -Is > "$P/ab.done"
