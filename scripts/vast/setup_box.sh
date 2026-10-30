#!/usr/bin/env bash
# Bring up a rented Vast.ai box (image nvidia/cuda:12.8.1-devel-ubuntu24.04): uv,
# Stockfish, then build ChessRL (the C++ extension and the fused CUDA network) for
# this GPU's architecture. Idempotent; safe to re-run.
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
export PATH="\$HOME/.local/bin:/usr/games:\$PATH"
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
