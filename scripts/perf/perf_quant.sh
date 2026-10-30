#!/usr/bin/env bash
# Quantization session (after scripts/perf/perf_session.sh, on the same box): FP16 vs INT8 vs FP8
# engines for a trained network and for bigger random-weight encoders (speed), and the accuracy
# of the quantized trained network (value and 400-simulation search on the gap study's 600
# positions, copied to /workspace/perf/gap_positions.json). Results in docs/PERF.md.
set -uo pipefail
source /workspace/env.sh
P=/workspace/perf
Q=$P/quant
mkdir -p "$Q"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$P/session.log"; }

log "== 7. quantization: ModelOpt in its own venv"
uv venv -q --python 3.12 /workspace/mo > "$Q/mo_install.log" 2>&1
uv pip install -q --python /workspace/mo "nvidia-modelopt[onnx]" >> "$Q/mo_install.log" 2>&1
log "modelopt: $(/workspace/mo/bin/python -c 'import modelopt; print(modelopt.__version__)' 2>&1 | tail -1)"
uv run python scripts/perf/quant_prep.py > "$Q/prep.log" 2>&1
log "prep: $(tr '\n' ';' < "$Q/prep.log")"
cp "${CKPT:-$P/step_00092500.pt}" "$Q/enc128x4.pt"

build() { # name, precision -> $Q/name_precision.trt
  local name=$1 prec=$2 onnx="$Q/$1.onnx" flags=(--fp16)
  case $prec in
    int8) onnx="$Q/${name}_int8.onnx"; flags=(--fp16 --int8) ;;
    fp8) onnx="$Q/${name}_fp8.onnx"; flags=(--fp16 --fp8) ;;
  esac
  [ -f "$onnx" ] || return 1
  trtexec --onnx="$onnx" "${flags[@]}" --minShapes=input:1x8x8x111 --optShapes=input:256x8x8x111 \
    --maxShapes=input:512x8x8x111 --shapes=input:256x8x8x111 --timingCacheFile="$Q/timing.cache" \
    --saveEngine="$Q/${name}_$prec.trt" > "$Q/build_${name}_$prec.log" 2>&1
}

for name in enc128x4 enc192x6 enc256x8 enc384x10; do
  uv run python -m src.export.to_onnx --ckpt_path "$Q/$name.pt" --onnx_path "$Q/$name.onnx" \
    --opset 19 > "$Q/onnx_$name.log" 2>&1 || { log "$name: ONNX export failed"; continue; }
  for mode in int8 fp8; do
    /workspace/mo/bin/python scripts/perf/quantize_onnx.py "$Q/$name.onnx" $mode "$Q/calib.npy" \
      "$Q/${name}_$mode.onnx" > "$Q/quant_${name}_$mode.log" 2>&1 \
      || log "$name $mode: quantization failed: $(tail -1 "$Q/quant_${name}_$mode.log")"
  done
  for prec in fp16 int8 fp8; do
    if ! build $name $prec; then
      log "$name $prec: no engine ($(tail -2 "$Q/build_${name}_$prec.log" 2>/dev/null | tr '\n' ' '))"
      continue
    fi
    line="$name $prec:"
    for B in 128 256 512; do
      trtexec --loadEngine="$Q/${name}_$prec.trt" --shapes=input:${B}x8x8x111 --useCudaGraph \
        --noDataTransfers --warmUp=500 --duration=6 > "$Q/thr_${name}_${prec}_$B.log" 2>&1
      qps=$(grep -oP 'Throughput: \K[0-9.]+' "$Q/thr_${name}_${prec}_$B.log" | tail -1)
      line="$line  b$B $(python3 -c "print(round(${qps:-0} * $B))") pos/s"
    done
    log "$line"
  done
done

log "== 8. accuracy of the quantized trained network"
args=("fp16=$Q/enc128x4_fp16.trt")
[ -f "$Q/enc128x4_int8.trt" ] && args+=("int8=$Q/enc128x4_int8.trt")
[ -f "$Q/enc128x4_fp8.trt" ] && args+=("fp8=$Q/enc128x4_fp8.trt")
uv run python scripts/perf/quant_accuracy.py "${args[@]}" > "$Q/accuracy.log" 2>&1
tee -a "$P/session.log" < "$Q/accuracy.log"
log "== quantization done"
date -Is > "$P/quant.done"
