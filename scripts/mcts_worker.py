"""MCTS worker — sits on stdin, returns chosen UCI moves per request.

Run with:
  PYTHONPATH=<root>:<root>/cpp ... .venv/bin/python scripts/mcts_worker.py \
      --engine <PATH> [--vlw N]

Protocol (line-delimited JSON):
  request : {"histories": [["e2e4", ...], ...]}
  reply   : {"actions":   ["e7e5", ...]}
Send empty line to flush; "QUIT" to exit.
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--vlw", type=int, default=0, help="0 = pass no virtual_loss_width (older API)")
ap.add_argument("--sims", type=int, default=800)
ap.add_argument("--cpuct", type=float, default=0.65)
args = ap.parse_args()

import _inference_cpp

sys.stderr.write(f"[worker] ready  engine={args.engine}  sims={args.sims}  "
                 f"vlw={'native' if args.vlw == 0 else args.vlw}\n"); sys.stderr.flush()

for line in sys.stdin:
    line = line.strip()
    if not line: continue
    if line == "QUIT": break
    req = json.loads(line)
    histories = req["histories"]
    kw = dict(
        histories=histories,
        fens=["" for _ in histories],
        simulations=args.sims,
        cpuct=args.cpuct,
        temperature=1.0,
        engine_path=args.engine,
        temperature_at_root=0.0,
        use_dirichlet=False,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
    )
    if args.vlw > 0:
        kw["virtual_loss_width"] = args.vlw
    res = _inference_cpp.small_mcts_search_batch_with_history(**kw)
    actions = [r["action"] for r in res]
    print(json.dumps({"actions": actions}), flush=True)
