#!/usr/bin/env bash
# CPU benchmark of the self-play search with a fake network (no GPU): prints one table row
# per (mode, threads, sharpness) and, with PHASES=1, where the time goes. Builds the
# binary if needed. Run from anywhere; it works from the repo root.
#
#   scripts/bench_cpu_search.sh                      # full + fast, 1 and 8 threads
#   SHARPNESS="2 3 4" THREADS=1 MODES=fast scripts/bench_cpu_search.sh
#
# Env: MODES (full fast), THREADS (1 8), SHARPNESS (3), REPEAT (3), CORES (0-7, taskset
# list), LIMIT (0 = all positions), PHASES (0), EXTRA (more bench_search arguments).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BIN=cpp/build/bench/bench_search
[ -x "$BIN" ] || scripts/build_cpu_bench.sh > /dev/null

MODES="${MODES:-full fast}"
THREADS="${THREADS:-1 8}"
SHARPNESS="${SHARPNESS:-3}"
REPEAT="${REPEAT:-3}"
CORES="${CORES:-0-7}"
LIMIT="${LIMIT:-0}"
PHASES="${PHASES:-0}"
EXTRA="${EXTRA:-}"

rows=()
for mode in $MODES; do
  for threads in $THREADS; do
    for sharp in $SHARPNESS; do
      # shellcheck disable=SC2086
      row=$(nice -n 10 taskset -c "$CORES" "$BIN" --mode "$mode" --threads "$threads" \
        --sharpness "$sharp" --repeat "$REPEAT" --limit "$LIMIT" $EXTRA)
      rows+=("$row")
      if [ "$PHASES" = 1 ]; then
        # shellcheck disable=SC2086
        CHESSRL_MCTS_PHASES=1 nice -n 10 taskset -c "$CORES" "$BIN" --mode "$mode" \
          --threads "$threads" --sharpness "$sharp" --repeat 1 --limit "$LIMIT" $EXTRA 2>&1 >/dev/null \
          | grep mcts-phases | tail -1 | sed "s/^/  $mode t=$threads s=$sharp /" >&2
      fi
    done
  done
done

printf '%s\n' "${rows[@]}" | python3 scripts/bench_cpu_table.py
