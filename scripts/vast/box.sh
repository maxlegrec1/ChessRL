#!/usr/bin/env bash
# Box-side helper, copied to /workspace/box.sh by setup_box.sh.
#   box.sh update                      fetch + hard-reset the branch, uv sync
#   box.sh run <config> <name> [gpu]   start a training run in tmux session run-<name>
#   box.sh resume <name> [gpu]         resume the newest run dir ending in _<name>
#   box.sh ladder <name> [gpu]         live ladder daemon for that run
#   box.sh dash                        dashboard on :8501
#   box.sh calibrate                   one-off anchor round robin
#   box.sh status                      sessions, GPU, newest log lines
#   box.sh stop <session>
set -euo pipefail
source /workspace/env.sh
RUNS=/workspace/ChessRL/artifacts/runs
cmd="${1:-status}"
shift || true

run_dir() { ls -d "$RUNS"/*_"$1" 2>/dev/null | sort | tail -1; }

case "$cmd" in
  update)
    branch="$(git rev-parse --abbrev-ref HEAD)"
    git fetch -q origin "$branch"
    git reset -q --hard "origin/$branch"
    git submodule update -q --init
    uv sync -q --group dev --group viz
    cp scripts/vast/box.sh /workspace/box.sh
    git log -1 --format='%h %s'
    ;;
  run)
    cfg="$1"; name="$2"; gpu="${3:-0}"
    tmux new -d -s "run-$name" "source /workspace/env.sh && CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.run --config $cfg --name $name 2>&1 | tee -a /workspace/run-$name.log"
    echo "started run-$name on GPU $gpu"
    ;;
  resume)
    name="$1"; gpu="${2:-0}"; dir="$(run_dir "$name")"
    tmux new -d -s "run-$name" "source /workspace/env.sh && CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.run --resume $dir 2>&1 | tee -a /workspace/run-$name.log"
    echo "resumed $dir"
    ;;
  ladder)
    name="$1"; gpu="${2:-0}"; dir="$(run_dir "$name")"
    tmux new -d -s "ladder-$name" "source /workspace/env.sh && CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/ladder_daemon.py --run-dir $dir 2>&1 | tee -a $dir/ladder.log"
    echo "ladder for $dir"
    ;;
  dash)
    tmux new -d -s dash "source /workspace/env.sh && uv run streamlit run scripts/dashboard.py --server.port 8501 --server.headless true"
    ;;
  calibrate)
    tmux new -d -s calib "source /workspace/env.sh && uv run python scripts/ladder_daemon.py --calibrate 2>&1 | tee /workspace/calib.log"
    ;;
  status)
    tmux ls 2>/dev/null || true
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
    uptime
    for d in "$RUNS"/*/; do
      [ -d "$d" ] || continue
      echo "== $(basename "$d")"
      tail -n 2 "$d"/selfplay.log 2>/dev/null | cut -c1-220 || true
      tail -n 1 "$d"/training.log 2>/dev/null | cut -c1-220 || true
    done
    ;;
  stop)
    tmux kill-session -t "$1"
    ;;
  *)
    echo "unknown command $cmd"; exit 2
    ;;
esac
