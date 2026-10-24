"""How often does the self-play search pick a different move when simply run twice?

Control for head_ab.py's Dirichlet test: runs the same search (400 sims, cpuct 0.65,
prior temperature 1.3, virtual-loss width 8) twice without noise, and twice with
noise, on the same positions, and reports how often the most-visited move differs.

  uv run python scripts/search_determinism.py --engine X.trt --positions positions.json [--n 500]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def search(engine: str, positions: list[list[str]], noise: bool, sims: int, vlw: int) -> list[dict]:
    import _inference_cpp

    return _inference_cpp.small_mcts_search_batch_with_history(
        histories=positions, fens=[""] * len(positions), simulations=sims, cpuct=0.65, temperature=1.3,
        engine_path=engine, temperature_at_root=0.0, use_dirichlet=noise, dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25, virtual_loss_width=vlw)


def best(res: list[dict]) -> list[str]:
    return [max(r["policy"], key=lambda mv: mv[1])[0] for r in res]


def root_e(res: list[dict]) -> np.ndarray:
    return np.array([r["value"][0] + 0.5 * r["value"][1] for r in res])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--positions", type=Path, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    pos = json.loads(a.positions.read_text())[: a.n]
    out = {"positions": len(pos)}
    for vlw in (8, 1):
        off1, off2 = search(a.engine, pos, False, a.sims, vlw), search(a.engine, pos, False, a.sims, vlw)
        on1, on2 = search(a.engine, pos, True, a.sims, vlw), search(a.engine, pos, True, a.sims, vlw)
        diff = lambda x, y: round(float(np.mean([i != j for i, j in zip(best(x), best(y), strict=True)])), 4)  # noqa: E731
        # Where noise changed the move: how much worse is it, by the noise-free search's
        # own values (expected score for the side to move; q_values are in [-1, 1])?
        losses = []
        for r_off, m_off, m_on in zip(off1, best(off1), best(on1), strict=True):
            if m_on != m_off and m_on in r_off["q_values"]:
                losses.append((r_off["q_values"][m_off] - r_off["q_values"][m_on]) / 2.0)
        losses = np.array(losses) if losses else np.zeros(1)
        out[f"vlw{vlw}"] = {
            "changed_move_value_loss_mean": round(float(losses.mean()), 4),
            "changed_move_value_loss_median": round(float(np.median(losses)), 4),
            "changed_move_loss_over_0.05": round(float((losses > 0.05).mean()), 4),
            "changed_move_loss_over_0.10": round(float((losses > 0.10).mean()), 4),
            "move_changed_off_vs_off": diff(off1, off2),
            "move_changed_on_vs_off": diff(on1, off1),
            "move_changed_on_vs_on": diff(on1, on2),
            "root_shift_off_vs_off_abs": round(float(np.abs(root_e(off1) - root_e(off2)).mean()), 4),
            "root_shift_on_vs_off_abs": round(float(np.abs(root_e(on1) - root_e(off1)).mean()), 4),
        }
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
