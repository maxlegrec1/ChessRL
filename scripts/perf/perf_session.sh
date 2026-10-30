#!/usr/bin/env bash
# GPU performance session on a rented box (scripts/vast), results in docs/PERF.md. After
# scripts/vast/setup_box.sh, copy a checkpoint to /workspace/perf/ (CKPT=... to choose it), then:
#   cd /workspace/ChessRL && nohup bash scripts/perf/perf_session.sh > /workspace/perf/session.out 2>&1 &
# Logs everything to /workspace/perf/; one summary line per measurement goes to session.log.
# scripts/perf/perf_quant.sh (FP16 / INT8 / FP8) can follow it or run alone.
set -uo pipefail
source /workspace/env.sh
P=/workspace/perf
CKPT="${CKPT:-$P/step_00092500.pt}"
mkdir -p "$P"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$P/session.log"; }
now() { date +%s.%N; }
since() { python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$1" "$(now)"; }
bench() { # label, then bench_selfplay.py overrides
  local label=$1
  shift
  uv run python scripts/bench_selfplay.py --engine "$P/net.trt" --duration-sec "${DUR:-90}" \
    --warmup-sec 15 --label "$label" --out "$P/bench_$label.json" "$@" > "$P/bench_$label.log" 2>&1
  log "selfplay $label: $(grep -o '[0-9.]* positions/sec.*' "$P/bench_$label.log" | head -1) |" \
    "$(grep -o 'inference: .*' "$P/bench_$label.log" | head -1)"
}
TRT=(trtexec --onnx="$P/net.onnx" --fp16 --minShapes=input:1x8x8x111 --optShapes=input:256x8x8x111
  --maxShapes=input:512x8x8x111 --shapes=input:256x8x8x111)

log "== 1. build and tests"
log "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); $(nproc) cores; $(lscpu | grep -m1 'Model name' | sed 's/  */ /g')"
(bash scripts/build_cpp_tests.sh && cpp/build/tests/cpp_tests) > "$P/cpp_tests.log" 2>&1
log "C++ tests: $(grep 'test cases' "$P/cpp_tests.log")"
uv run pytest -q tests/test_inference_cpp.py tests/test_encoder_parity.py tests/test_packed_parity.py \
  tests/test_packed_gpu_parity.py > "$P/pytest_gpu.log" 2>&1
log "GPU tests: $(tail -1 "$P/pytest_gpu.log")"

log "== 2. engine build"
t0=$(now)
uv run python -m src.export.to_onnx --ckpt_path "$CKPT" --onnx_path "$P/net.onnx" > "$P/onnx.log" 2>&1
log "ONNX export: $(since "$t0") s"
t0=$(now); "${TRT[@]}" --saveEngine="$P/net.trt" > "$P/build_cold.log" 2>&1
log "trtexec build, as today (no timing cache): $(since "$t0") s"
t0=$(now); "${TRT[@]}" --saveEngine="$P/net_tc1.trt" --timingCacheFile="$P/timing.cache" > "$P/build_tc1.log" 2>&1
log "trtexec build, filling a timing cache: $(since "$t0") s"
t0=$(now); "${TRT[@]}" --saveEngine="$P/net_tc2.trt" --timingCacheFile="$P/timing.cache" > "$P/build_tc2.log" 2>&1
log "trtexec build, reusing the timing cache: $(since "$t0") s"

log "== 3. engine alone: throughput by batch size (CUDA graphs, no host transfers)"
for B in 32 64 128 256 512; do
  trtexec --loadEngine="$P/net.trt" --shapes=input:${B}x8x8x111 --useCudaGraph --noDataTransfers \
    --warmUp=1000 --duration=10 > "$P/thr_$B.log" 2>&1
  qps=$(grep -oP 'Throughput: \K[0-9.]+' "$P/thr_$B.log" | tail -1)
  ms=$(grep -oP 'GPU Compute Time: .*?mean = \K[0-9.]+' "$P/thr_$B.log" | tail -1)
  log "batch $B: ${qps} batches/s, ${ms} ms GPU per batch -> $(python3 -c "print(round(${qps:-0} * $B))") positions/s"
done
trtexec --loadEngine="$P/net.trt" --shapes=input:128x8x8x111 --dumpProfile --separateProfileRun \
  --exportProfile="$P/profile_128.json" --duration=5 > "$P/profile_128.log" 2>&1
log "per-layer profile at batch 128 -> profile_128.json"

log "== 4. self-play (recipe: 1024 boards, 128 threads, 4 inference workers)"
bench recipe
DUR=45 CHESSRL_MCTS_PHASES=1 bench phases
log "phases (last steps): $(grep mcts-phases "$P/bench_phases.log" | tail -2 | tr '\n' ' ')"
CHESSRL_INFER_WORKERS=2 bench workers2
bench threads64 --threads 64
bench threads256 --threads 256
CHESSRL_GRAPH_BUCKETS=32,64,128,256,512 bench threads512 --threads 512

log "== 5. contention"
(sleep 20; t0=$(now); "${TRT[@]}" --saveEngine="$P/net_bg.trt" > "$P/build_bg.log" 2>&1
 log "  background build (no cache) took $(since "$t0") s") &
DUR=150 bench during_build
wait
(sleep 20; t0=$(now); "${TRT[@]}" --saveEngine="$P/net_bg2.trt" --timingCacheFile="$P/timing.cache" > "$P/build_bg2.log" 2>&1
 log "  background build (timing cache) took $(since "$t0") s") &
DUR=150 bench during_build_cached
wait
uv run python scripts/perf/gpu_load.py --ckpt "$CKPT" --seconds 115 --duty 0.15 > "$P/load15.log" 2>&1 &
bench trainer15
wait
log "  $(cat "$P/load15.log" | tail -1)"
uv run python scripts/perf/gpu_load.py --ckpt "$CKPT" --seconds 115 --duty 1.0 > "$P/load100.log" 2>&1 &
bench trainer100
wait
log "  $(cat "$P/load100.log" | tail -1)"
if nvidia-cuda-mps-control -d > "$P/mps.log" 2>&1; then
  log "MPS daemon started"
  bench mps_alone
  uv run python scripts/perf/gpu_load.py --ckpt "$CKPT" --seconds 115 --duty 0.15 > "$P/load15_mps.log" 2>&1 &
  bench mps_trainer15
  wait
  log "  $(cat "$P/load15_mps.log" | tail -1)"
  echo quit | nvidia-cuda-mps-control
else
  log "MPS could not start: $(tail -1 "$P/mps.log")"
fi

log "== 6. kernel timeline (nsys)"
command -v nsys > /dev/null || apt-get install -y -qq nsight-systems-cli > "$P/nsys_install.log" 2>&1
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems*/bin/nsys 2>/dev/null | head -1)
if [ -n "$NSYS" ]; then
  "$NSYS" profile --trace=cuda,osrt --sample=none --delay=20 --duration=20 -f true -o "$P/selfplay" \
    uv run python scripts/bench_selfplay.py --engine "$P/net.trt" --duration-sec 30 --warmup-sec 10 \
    --label nsys --out "$P/bench_nsys.json" > "$P/nsys.log" 2>&1
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv -o "$P/kern" "$P/selfplay.nsys-rep" > "$P/nsys_stats.log" 2>&1
  log "nsys: $(ls "$P"/kern* 2>/dev/null | tr '\n' ' ')"
else
  log "nsys not available"
fi
log "== done"
date -Is > "$P/session.done"
