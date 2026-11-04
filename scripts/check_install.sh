#!/usr/bin/env bash
# scripts/check_install.sh — pre-flight check for ChessRL.
#
# Validates host dependencies before `uv sync`. Run once on a fresh machine;
# exits non-zero on the first failure with a remediation hint.
set -u

red() { printf '\033[31m%s\033[0m\n' "$*"; }
grn() { printf '\033[32m%s\033[0m\n' "$*"; }
ylw() { printf '\033[33m%s\033[0m\n' "$*"; }

fail() {
  red "✗ $1"
  echo "  fix: $2"
  exit 1
}
ok()    { grn "✓ $1"; }
note()  { ylw "ℹ $1"; }

echo "ChessRL install pre-flight"
echo "=========================="

# --- uv ---------------------------------------------------------------------
# uv handles Python 3.13 provisioning automatically (`uv sync` will install it
# if missing), so we don't separately check for a system python3.
if ! command -v uv >/dev/null 2>&1; then
  fail "uv not on PATH" \
       "curl -LsSf https://astral.sh/uv/install.sh | sh   (then restart shell)"
fi
ok "uv $(uv --version | awk '{print $2}')"

# --- NVIDIA driver / GPU ----------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not on PATH" \
       "install NVIDIA driver (https://www.nvidia.com/Download/index.aspx)"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [ -z "$GPU_NAME" ]; then
  fail "no CUDA-visible GPUs" \
       "check 'nvidia-smi'; ensure driver is loaded and CUDA_VISIBLE_DEVICES is unset"
fi
ok "GPU: $GPU_NAME"

# --- CUDA toolkit / nvcc ----------------------------------------------------
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
if [ ! -x "$CUDA_ROOT/bin/nvcc" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    note "nvcc found on PATH but not in CUDA_ROOT=$CUDA_ROOT/bin"
    note "set CUDA_ROOT to the toolkit root, e.g. export CUDA_ROOT=/usr/local/cuda-12.8"
  else
    fail "nvcc not at $CUDA_ROOT/bin/nvcc" \
         "install CUDA Toolkit 12+ and either symlink /usr/local/cuda or set CUDA_ROOT"
  fi
fi
if [ -x "$CUDA_ROOT/bin/nvcc" ]; then
  CUDA_VER=$("$CUDA_ROOT/bin/nvcc" --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
  ok "nvcc $CUDA_VER  (CUDA_ROOT=$CUDA_ROOT)"
fi

# --- Stockfish (the rating ladder's reference opponents; training runs without it) ---
SF=$(command -v stockfish || { [ -x /usr/games/stockfish ] && echo /usr/games/stockfish; } || true)
if [ -n "$SF" ]; then
  ok "Stockfish at $SF"
else
  note "Stockfish not found; the rating ladder needs it ('apt install stockfish', version 16 calibrated)"
fi

# --- CHESSRL_CUDA_ARCH sanity ----------------------------------------------
ARCH=${CHESSRL_CUDA_ARCH:-sm_90}
case "$ARCH" in
  sm_[0-9]*)
    ok "CHESSRL_CUDA_ARCH=$ARCH"
    ;;
  *)
    fail "CHESSRL_CUDA_ARCH=$ARCH is malformed" \
         "use sm_90 (H200/Hopper), sm_89 (4090/Ada), sm_86 (3090/Ampere), sm_80 (A100)"
    ;;
esac

echo
grn "All checks passed. Next: 'uv sync' to build the C++ extension."
