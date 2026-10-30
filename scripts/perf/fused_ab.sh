#!/usr/bin/env bash
# Fused network vs TensorRT on a rented GPU box (after scripts/vast/setup_box.sh), results
# in docs/PERF.md: parity, the kernel alone, then self-play speed with the recipe and with
# tree reuse, on each backend, 60 s of warm-up and 240 s measured each (ROUNDS=2 repeats
# the four). Needs a checkpoint at /workspace/perf/step_00092500.pt (CKPT=... to choose)
# and the TensorRT engine tree_reuse_ab.sh builds (net.trt, rebuilt if missing).
#   cd /workspace/ChessRL && nohup bash scripts/perf/fused_ab.sh > /workspace/perf/fused.out 2>&1 &
set -uo pipefail
source /workspace/env.sh
P=/workspace/perf
CKPT="${CKPT:-$P/step_00092500.pt}"
ROUNDS="${ROUNDS:-1}"
mkdir -p "$P"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$P/fused.log"; }

log "commit $(git log --oneline -1); $(nvidia-smi --query-gpu=name --format=csv,noheader); $(lscpu | grep -m1 'Model name' | sed 's/  */ /g')"
uv run pytest -q tests/test_cnet_export.py tests/test_inference_cpp.py tests/test_packed_gpu_parity.py \
  > "$P/pytest_fused.log" 2>&1
log "tests: $(tail -1 "$P/pytest_fused.log")"
uv run python scripts/net_parity.py "$CKPT" --n 256 > "$P/net_parity.log" 2>&1
log "parity: $(tail -1 "$P/net_parity.log")"

uv run python -m src.export.to_cnet --ckpt_path "$CKPT" --out "$P/net.cnet" > "$P/cnet.log" 2>&1
if [ ! -f "$P/net.trt" ]; then
  uv run python -m src.export.to_onnx --ckpt_path "$CKPT" --onnx_path "$P/net.onnx" > "$P/onnx.log" 2>&1
  trtexec --onnx="$P/net.onnx" --saveEngine="$P/net.trt" --fp16 --minShapes=input:1x8x8x111 \
    --optShapes=input:256x8x8x111 --maxShapes=input:512x8x8x111 --shapes=input:256x8x8x111 \
    --timingCacheFile="$P/timing.cache" > "$P/build.log" 2>&1
fi
uv run python -c "
import _inference_cpp as c
for s in (1, 4):
    print('kernel alone, streams', s, ' '.join(f'{b}:{c.fused_benchmark(\"$P/net.cnet\", b, 300, s) / 1e3:.0f}k' for b in (32, 64, 128, 256, 512)))
" > "$P/kernel.log" 2>&1
log "$(tr '\n' ';' < "$P/kernel.log")"

printf 'extends: ../ChessRL/src/train/config.yaml\nselfplay:\n  tree_reuse: true\n' > "$P/reuse.yaml"
for round in $(seq 1 "$ROUNDS"); do
  for setting in trt_recipe fused_recipe trt_reuse fused_reuse; do
    cfg=src/train/config.yaml
    case $setting in *reuse) cfg="$P/reuse.yaml" ;; esac
    engine="$P/net.trt"
    case $setting in fused*) engine="$P/net.cnet" ;; esac
    label="${setting}_$round"
    vmstat 30 > "$P/vmstat_$label.log" &
    vm=$!
    uv run python scripts/bench_selfplay.py --engine "$engine" --config "$cfg" --warmup-sec 60 \
      --duration-sec 240 --label "$label" --out "$P/fused_$label.json" > "$P/fused_$label.log" 2>&1
    kill $vm
    cpu=$(awk 'NR > 3 {busy += 100 - $15; n++} END {if (n) printf "%.0f", busy / n}' "$P/vmstat_$label.log")
    log "$label: $(grep -o '[0-9.]* positions/sec.*' "$P/fused_$label.log" | head -1) | CPU busy ${cpu}% |" \
      "$(grep -o 'inference: .*' "$P/fused_$label.log" | head -1)"
  done
done
log "== done"
date -Is > "$P/fused.done"
