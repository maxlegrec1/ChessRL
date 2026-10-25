"""Check that a zero-simulation search samples the raw policy at the requested temperature.

Self-play's quick_policy_temperature relies on it: with simulations=0, prior temperature
1 and temperature_at_root=T, the engine should draw moves from softmax(logit / T). This
samples it many times on a few positions and compares the frequencies with the PyTorch
network's policy (total variation distance; ~0.02-0.04 is sampling noise at 4,000 draws).

  uv run python scripts/check_policy_sampling.py --ckpt X.pt --engine X.trt --positions positions.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.engine import Net  # noqa: E402


def main() -> None:
    import _inference_cpp
    import chess

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--positions", type=Path, required=True, help="policy_gap.py positions JSON")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--temp", type=float, default=0.7)
    a = p.parse_args()

    rows = [r for r in json.loads(a.positions.read_text()) if r["phase"] == "opening"][: a.n]
    net = Net(a.ckpt, threads=4)
    worst = 0.0
    for r in rows:
        board = chess.Board()
        for u in r["moves"]:
            board.push_uci(u)
        priors = {m.uci(): pr for m, pr in net.evaluate([board])[0][0].items()}
        w = {m: pr ** (1.0 / a.temp) for m, pr in priors.items()}
        z = sum(w.values())
        expected = {m: v / z for m, v in w.items()}
        res = _inference_cpp.small_mcts_search_batch_with_history(
            histories=[r["moves"]] * a.draws, fens=[""] * a.draws, simulations=0, cpuct=0.65, temperature=1.0,
            engine_path=a.engine, temperature_at_root=a.temp, use_dirichlet=False, virtual_loss_width=8)
        counts = collections.Counter(x["action"] for x in res)
        tv = 0.5 * sum(abs(counts.get(m, 0) / a.draws - expected.get(m, 0.0)) for m in set(counts) | set(expected))
        worst = max(worst, tv)
        top = sorted(expected.items(), key=lambda kv: -kv[1])[:4]
        print(f"ply {len(r['moves']):3d}  TV {tv:.3f}  " + "  ".join(
            f"{m} exp {v:.3f} got {counts.get(m, 0) / a.draws:.3f}" for m, v in top))
    print(f"worst total variation: {worst:.3f}", "OK" if worst < 0.06 else "MISMATCH")


if __name__ == "__main__":
    main()
