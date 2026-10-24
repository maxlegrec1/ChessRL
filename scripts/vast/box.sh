#!/usr/bin/env bash
# Box-side helper, copied to /workspace/box.sh by setup_box.sh.
#   box.sh update                      fetch + hard-reset the branch, uv sync
#   box.sh run <config> <name> [gpu]   start a training run in tmux session run-<name>
#   box.sh resume <name> [gpu]         resume the newest run dir ending in _<name>
#   box.sh ladder <name> [gpu]         live ladder daemon for that run
#   box.sh keep <name> [gpu]           resume run-<name> if it dies (up to 5 times)
#   box.sh dash                        dashboard on :8501
#   box.sh calibrate                   one-off anchor round robin
#   box.sh status                      sessions, GPU, newest log lines
#   box.sh stop <session>
set -euo pipefail
source /workspace/env.sh
RUNS=/workspace/ChessRL/artifacts/runs
# CPUS=0-31 box.sh run ... pins the run (and its ladder) to those CPUs so every
# arm gets the same core count whatever box it lands on.
PIN=""
if [ -n "${CPUS:-}" ]; then
  # Containers limited to their own cpuset reject other ranges: run unpinned then.
  if taskset -c "$CPUS" true 2>/dev/null; then PIN="taskset -c $CPUS"; else echo "CPUS=$CPUS not allowed here; not pinning"; fi
fi
# EXTRA_ENV="CHESSRL_CUDA_GRAPHS=0 ..." is prepended to the run's command (tmux
# sessions take the tmux server's environment, not the caller's).
XENV="${EXTRA_ENV:-}"
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
    git log -1 --format='%h %s'
    # Atomic rename: bash is still reading this very file.
    cp scripts/vast/box.sh /workspace/box.sh.new && mv /workspace/box.sh.new /workspace/box.sh
    ;;
  run)
    cfg="$1"; name="$2"; gpu="${3:-0}"
    tmux new -d -s "run-$name" "source /workspace/env.sh && CUDA_VISIBLE_DEVICES=$gpu $XENV $PIN uv run python -m src.run --config $cfg --name $name 2>&1 | tee -a /workspace/run-$name.log"
    echo "started run-$name on GPU $gpu"
    ;;
  resume)
    name="$1"; gpu="${2:-0}"; dir="$(run_dir "$name")"
    tmux new -d -s "run-$name" "source /workspace/env.sh && CUDA_VISIBLE_DEVICES=$gpu $XENV $PIN uv run python -m src.run --resume $dir 2>&1 | tee -a /workspace/run-$name.log"
    echo "resumed $dir"
    ;;
  resume-with)
    # resume-with <name> <config> [gpu]: continue a run under a different config.
    name="$1"; cfg="$2"; gpu="${3:-0}"; dir="$(run_dir "$name")"
    tmux new -d -s "run-$name" "source /workspace/env.sh && CUDA_VISIBLE_DEVICES=$gpu $XENV $PIN uv run python -m src.run --resume $dir --config $cfg 2>&1 | tee -a /workspace/run-$name.log"
    echo "resumed $dir with $cfg"
    ;;
  keep)
    # keep <name> [gpu]: if run-<name> dies (self-play crash -> the launcher stops
    # both halves), resume it from its newest checkpoint; at most 5 times. Takes
    # the same CPUS / EXTRA_ENV as the run.
    name="$1"; gpu="${2:-0}"
    tmux new -d -s "keep-$name" "for i in 1 2 3 4 5; do while tmux has-session -t run-$name 2>/dev/null; do sleep 60; done; echo \"[keep] \$(date -u +%H:%M) run-$name gone; resume \$i\" >> /workspace/keep-$name.log; CPUS='${CPUS:-}' EXTRA_ENV='$XENV' bash /workspace/box.sh resume $name $gpu >> /workspace/keep-$name.log 2>&1; sleep 120; done"
    echo "keep-alive for run-$name"
    ;;
  ladder)
    # Next to a 6-8 layer encoder's self-play + trainer (~20 GB) the default
    # runner (one TensorRT context per worker per CUDA-graph bucket, batch 256:
    # 400 MB each) no longer fits. The ladder needs no throughput: batch 64,
    # 2 workers, no graphs. The loop restarts the daemon after a crash (it
    # resumes from ladder.jsonl).
    name="$1"; gpu="${2:-0}"; dir="$(run_dir "$name")"
    tmux new -d -s "ladder-$name" "source /workspace/env.sh && while true; do CUDA_VISIBLE_DEVICES=$gpu CHESSRL_MCTS_BATCH_SIZE=${LADDER_BATCH:-64} CHESSRL_CUDA_GRAPHS=0 CHESSRL_INFER_WORKERS=2 $PIN uv run python scripts/ladder_daemon.py --run-dir $dir 2>&1 | tee -a $dir/ladder.log; echo '[box] ladder exited; restarting in 60 s' | tee -a $dir/ladder.log; sleep 60; done"
    echo "ladder for $dir"
    ;;
  dash)
    tmux new -d -s dash "source /workspace/env.sh && CHESSRL_RUNS_ROOT=$RUNS:/workspace/ChessRL-main/artifacts/runs uv run streamlit run scripts/dashboard.py --server.port 8501 --server.headless true"
    ;;
  main-setup)
    # The untouched original code (main@128893c) for the baseline arm.
    if [ ! -d /workspace/ChessRL-main/.git ]; then
      git clone -q https://github.com/maxlegrec1/ChessRL.git /workspace/ChessRL-main
    fi
    cd /workspace/ChessRL-main
    git checkout -q 128893c
    git submodule update -q --init
    uv sync -q
    uv run python -c "import _inference_cpp; print('main build ok')"
    ;;
  main-run)
    # main-run <name> [gpu]: main's own launcher + config, W&B offline; the
    # metrics adapter and the ladder daemon come from the next-round checkout.
    name="$1"; gpu="${2:-0}"
    tmux new -d -s "run-$name" "source /workspace/env.sh && cd /workspace/ChessRL-main && CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline $PIN uv run python -m src.run --name $name 2>&1 | tee -a /workspace/run-$name.log"
    sleep 90
    dir="$(ls -d /workspace/ChessRL-main/artifacts/runs/*_"$name" | sort | tail -1)"
    tmux new -d -s "legacy-$name" "source /workspace/env.sh && uv run python scripts/legacy_metrics.py --run-dir $dir"
    tmux new -d -s "ladder-$name" "source /workspace/env.sh && CUDA_VISIBLE_DEVICES=$gpu CHESSRL_MCTS_BATCH_SIZE=256 $PIN uv run python scripts/ladder_daemon.py --run-dir $dir 2>&1 | tee -a $dir/ladder.log"
    echo "main run in $dir"
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
    # Ctrl-C lets src.run forward SIGINT to its children (they live in their own
    # process groups, so killing the tmux session alone orphans them).
    session="$1"
    # A deliberate stop must not be undone by the keep-alive loop.
    case "$session" in run-*) tmux kill-session -t "keep-${session#run-}" 2>/dev/null || true ;; esac
    tmux send-keys -t "$session" C-c 2>/dev/null || true
    for _ in $(seq 1 30); do
      tmux has-session -t "$session" 2>/dev/null || break
      sleep 1
    done
    tmux kill-session -t "$session" 2>/dev/null || true
    case "$session" in
      run-*)
        dir="$(run_dir "${session#run-}")"
        [ -n "$dir" ] && pkill -f "$dir/config.yaml" || true
        ;;
    esac
    echo "stopped $session"
    ;;
  *)
    echo "unknown command $cmd"; exit 2
    ;;
esac
