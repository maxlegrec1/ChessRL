"""Format bench_search JSON lines (stdin) as the table printed by bench_cpu_search.sh."""

import json
import sys

rows = [json.loads(line) for line in sys.stdin if line.strip()]
print(
    f"{'mode':<5} {'thr':>3} {'sharp':>5} {'eff mv':>6} {'evals':>9} {'wall s':>7} "
    f"{'evals/s':>9} {'search us/eval':>14}  det  digest"
)
for d in rows:
    print(
        f"{d['mode']:<5} {d['threads']:>3} {d['sharpness']:>5.1f} {d['effective_moves']:>6.1f} "
        f"{d['evals']:>9} {d['wall_s']:>7.2f} {d['evals_per_s']:>9.0f} "
        f"{d['search_us_per_eval']:>14.2f}  {'yes' if d['deterministic'] else 'NO '}  {d['digest']}"
    )
