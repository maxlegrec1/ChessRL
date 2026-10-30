#!/usr/bin/env bash
# Tree reuse A/B on a rented GPU box (after scripts/vast/setup_box.sh), results in docs/PERF.md:
# self-play speed with the recipe, then with selfplay.tree_reuse on, alternated twice, each
# 60 s of warm-up (games in progress, kept trees) and 240 s measured. Needs a checkpoint at
# /workspace/perf/step_00092500.pt (CKPT=... to choose it).
#   cd /workspace/ChessRL && nohup bash scripts/perf/tree_reuse_ab.sh > /workspace/perf/ab.out 2>&1 &
set -uo pipefail
source /workspace/env.sh
P=/workspace/perf
CKPT="${CKPT:-$P/step_00092500.pt}"
mkdir -p "$P"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$P/ab.log"; }

log "commit $(git log --oneline -1); $(nvidia-smi --query-gpu=name --format=csv,noheader); $(lscpu | grep -m1 'Model name' | sed 's/  */ /g')"
(bash scripts/build_cpp_tests.sh && cpp/build/tests/cpp_tests) > "$P/cpp_tests.log" 2>&1
log "C++ tests: $(grep 'test cases' "$P/cpp_tests.log")"
uv run pytest -q tests/test_inference_cpp.py tests/test_encoder_parity.py tests/test_packed_parity.py \
  tests/test_packed_gpu_parity.py > "$P/pytest_gpu.log" 2>&1
log "GPU tests: $(tail -1 "$P/pytest_gpu.log")"

if [ ! -f "$P/net.trt" ]; then
  uv run python -m src.export.to_onnx --ckpt_path "$CKPT" --onnx_path "$P/net.onnx" > "$P/onnx.log" 2>&1
  trtexec --onnx="$P/net.onnx" --saveEngine="$P/net.trt" --fp16 --minShapes=input:1x8x8x111 \
    --optShapes=input:256x8x8x111 --maxShapes=input:512x8x8x111 --shapes=input:256x8x8x111 \
    --timingCacheFile="$P/timing.cache" > "$P/build.log" 2>&1
  log "engine built: $(ls -la "$P/net.trt" | awk '{print $5}') bytes"
fi

printf 'extends: ../ChessRL/src/train/config.yaml\nselfplay:\n  tree_reuse: true\n' > "$P/reuse.yaml"
for round in 1 2; do
  for setting in recipe reuse; do
    cfg=src/train/config.yaml
    [ "$setting" = reuse ] && cfg="$P/reuse.yaml"
    label="${setting}_$round"
    uv run python scripts/bench_selfplay.py --engine "$P/net.trt" --config "$cfg" --warmup-sec 60 \
      --duration-sec 240 --label "$label" --out "$P/ab_$label.json" > "$P/ab_$label.log" 2>&1
    log "$label: $(grep -o '[0-9.]* positions/sec.*' "$P/ab_$label.log" | head -1) |" \
      "$(grep -o 'inference: .*' "$P/ab_$label.log" | head -1)"
  done
done
log "== done"
date -Is > "$P/ab.done"
