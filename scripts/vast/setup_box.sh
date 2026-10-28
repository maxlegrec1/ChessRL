#!/usr/bin/env bash
# Bring up a rented Vast.ai box (image nvidia/cuda:12.8.1-devel-ubuntu24.04):
# TensorRT dev + trtexec from NVIDIA's apt repo, uv, Stockfish, then build
# ChessRL for this GPU's architecture. Idempotent; safe to re-run.
#   bash setup_box.sh [branch]
set -euo pipefail
BRANCH="${1:-main}"
REPO_URL="https://github.com/maxlegrec1/ChessRL.git"
WORK=/workspace
REPO="$WORK/ChessRL"
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$WORK"
log() { echo "[setup $(date +%H:%M:%S)] $*"; }

log "apt packages"
apt-get update -qq
apt-get install -y -qq --no-install-recommends git rsync tmux curl ca-certificates \
    build-essential stockfish htop >/dev/null

CUDA_MAJOR="$(nvcc --version | grep -oE 'release [0-9]+' | awk '{print $2}')"
if ! command -v trtexec >/dev/null 2>&1 && [ ! -x /usr/src/tensorrt/bin/trtexec ]; then
  # Pin the TensorRT 10 series ChessRL was built against (the repo's default is
  # now TensorRT 11, with soname-11 packages). Override with TRT_SERIES=10.16.
  TRT_SERIES="${TRT_SERIES:-10.14}"
  log "TensorRT ${TRT_SERIES} for CUDA ${CUDA_MAJOR}.x"
  TRT_VER="$(apt-cache madison libnvinfer-dev | awk '{print $3}' | grep "^${TRT_SERIES}\." \
             | grep "+cuda${CUDA_MAJOR}\." | sort -V | tail -1)"
  [ -n "$TRT_VER" ] || { echo "no TensorRT ${TRT_SERIES} build for CUDA ${CUDA_MAJOR} in apt"; exit 1; }
  PKGS=""
  for p in libnvinfer10 libnvinfer-dev libnvinfer-headers-dev libnvinfer-plugin10 libnvinfer-plugin-dev \
           libnvinfer-headers-plugin-dev libnvonnxparsers10 libnvonnxparsers-dev libnvinfer-bin \
           libnvinfer-dispatch10 libnvinfer-lean10 libnvinfer-vc-plugin10; do
    if apt-cache madison "$p" | awk '{print $3}' | grep -qx "$TRT_VER"; then PKGS="$PKGS $p=$TRT_VER"; fi
  done
  apt-get install -y -qq --no-install-recommends $PKGS >/dev/null
fi
export PATH="/usr/src/tensorrt/bin:$PATH"
trtexec --help >/dev/null 2>&1 || { echo "trtexec missing"; exit 1; }

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  log "uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"

log "repo @ $BRANCH"
if [ ! -d "$REPO/.git" ]; then
  git clone -q --branch "$BRANCH" "$REPO_URL" "$REPO"
fi
cd "$REPO"
git fetch -q origin "$BRANCH"
git checkout -q "$BRANCH"
git reset -q --hard "origin/$BRANCH"
git submodule update -q --init

CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
export CHESSRL_CUDA_ARCH="sm_${CAP}"
export CUDA_ROOT=/usr/local/cuda
log "uv sync (CHESSRL_CUDA_ARCH=$CHESSRL_CUDA_ARCH)"
uv sync -q --group dev --group viz --reinstall-package chessrl-inference-cpp

cat > "$WORK/env.sh" <<EOF
export PATH="/usr/src/tensorrt/bin:\$HOME/.local/bin:/usr/games:\$PATH"
export CHESSRL_CUDA_ARCH=$CHESSRL_CUDA_ARCH
export CUDA_ROOT=/usr/local/cuda
cd $REPO
EOF
grep -q 'workspace/env.sh' "$HOME/.bashrc" 2>/dev/null || echo 'source /workspace/env.sh' >> "$HOME/.bashrc"
cp scripts/vast/box.sh "$WORK/box.sh"

log "import check"
uv run python -c "import _inference_cpp, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
date -Is > "$WORK/setup.done"
log "done"
