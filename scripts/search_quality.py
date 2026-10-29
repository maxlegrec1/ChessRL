"""Search quality per network evaluation: self-play's PUCT search against Gumbel root search,
judged by Stockfish on the positions of the policy -> search gap study (docs/NEXT_ROUND.md).

For each position and search setting, the training target (PUCT: the visit counts; Gumbel:
the improved policy) and the move played are scored with Stockfish's expected score for every
legal move (cached in gap_positions.json): expected loss when sampling a move from the target,
share of sampled moves losing more than 0.10 / 0.30, loss of the target's favourite move and
of the move played (Gumbel plays a sampled survivor, so it explores), the target's effective
number of moves, and the target's share on the move played (for PUCT, the share of the visits
a reused tree would keep). Same scoring as the gap study, one CPU worker per core.
--dump writes every position's target and move.

  uv run python scripts/search_quality.py --ckpt step_00092500.pt --out search_quality.jsonl
  uv run python scripts/search_quality.py --ckpt X.pt --configs puct:400 gumbel:32 --limit 100

Settings:
  policy         the raw policy (T = 1); the move played is its favourite
  puct:N         self-play's search: prior temperature 1.3, cpuct 0.65, virtual-loss batches
                 of 8, moves-left tiebreak, no root noise; plays the most visited move
  puctnoise:N    the same with self-play's Dirichlet root noise (alpha 0.3, epsilon 0.25), as
                 on its full searches
  gumbel:N       Gumbel root search, 16 candidates, the paper's rule below the root
  gumbelpuct:N   the same with PUCT (prior temperature 1) below the root
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.engine import Net, search  # noqa: E402
from src.analysis.gumbel import gumbel_search  # noqa: E402

DEFAULT_POSITIONS = Path.home() / "chessrl-runs/runC-open0/gap_positions.json"
DEFAULT_CONFIGS = [
    "policy",
    "puct:50",
    "puct:100",
    "puct:200",
    "puct:400",
    "puctnoise:400",
    "gumbel:8",
    "gumbel:16",
    "gumbel:32",
    "gumbel:64",
    "gumbel:128",
    "gumbelpuct:32",
]


class CountingNet:
    """A Net that counts the positions it evaluates."""

    def __init__(self, net: Net) -> None:
        self.net, self.count = net, 0

    def evaluate(self, boards):
        self.count += len(boards)
        return self.net.evaluate(boards)


_net: CountingNet | None = None


def _init(ckpt: str) -> None:
    global _net
    _net = CountingNet(Net(ckpt, threads=1))


def board_of(moves: list[str]) -> chess.Board:
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    return b


def run_config(config: str, board: chess.Board, rng: np.random.Generator) -> dict:
    """The setting's training target (by UCI), the move it plays, and its cost."""
    assert _net is not None
    kind, _, n = config.partition(":")
    sims = int(n) if n else 0
    before, t0 = _net.count, time.perf_counter()
    if kind == "policy":
        ((priors, _, _),) = _net.evaluate([board])
        target = {m.uci(): p for m, p in priors.items()}
        played = max(target, key=target.get)
    elif kind in ("puct", "puctnoise"):
        noise = (0.3, 0.25, rng) if kind == "puctnoise" else None
        root = search(
            _net, board, sims, cpuct=0.65, batch=8, mlh_eps=0.05, prior_temperature=1.3,
            root_noise=noise,
        )["root"]  # fmt: skip
        visits = {m.uci(): c.n for m, c in root.children.items() if c.n}
        total = sum(visits.values()) or 1
        target = {m: v / total for m, v in visits.items()}
        played = max(target, key=target.get)
    elif kind in ("gumbel", "gumbelpuct"):
        res = gumbel_search(
            _net, board, sims, rng=rng, interior="gumbel" if kind == "gumbel" else "puct"
        )
        target = {m.uci(): p for m, p in res["improved"].items()}
        played = res["action"].uci()
    else:
        raise ValueError(f"unknown setting {config}")
    return {
        "target": target,
        "played": played,
        "evals": _net.count - before,
        "seconds": time.perf_counter() - t0,
    }


def work(job: tuple[int, list[str], list[str], int]) -> list[dict]:
    index, moves, configs, seed = job
    board = board_of(moves)
    out = []
    for config in configs:
        rng = np.random.default_rng([seed, index, zlib.crc32(config.encode())])
        out.append({"index": index, "config": config, **run_config(config, board, rng)})
    return out


def summarize(rows: list[dict], scores: list[dict[str, float]]) -> dict:
    """The gap study's target metrics, plus the move played and the cost."""
    if not rows:
        return {"positions": 0}
    exp_loss, blunder, big, near, ent, top_loss, top_blunder = ([] for _ in range(7))
    played_loss, played_blunder, played_share, evals, secs = ([] for _ in range(5))
    for r, s in zip(rows, scores, strict=True):
        best = max(s.values())
        loss = {m: best - v for m, v in s.items()}
        tot = sum(p for m, p in r["target"].items() if m in s) or 1.0
        probs = {m: p / tot for m, p in r["target"].items() if m in s}
        exp_loss.append(sum(p * loss[m] for m, p in probs.items()))
        blunder.append(sum(p for m, p in probs.items() if loss[m] > 0.10))
        big.append(sum(p for m, p in probs.items() if loss[m] > 0.30))
        near.append(sum(p for m, p in probs.items() if loss[m] < 0.02))
        ent.append(-sum(p * math.log(p) for p in probs.values() if p > 0))
        top = max(probs, key=probs.get)
        top_loss.append(loss[top])
        top_blunder.append(float(loss[top] > 0.10))
        played_share.append(probs.get(r["played"], 0.0))
        played_loss.append(loss[r["played"]])
        played_blunder.append(float(loss[r["played"]] > 0.10))
        evals.append(r["evals"])
        secs.append(r["seconds"])

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4)

    return {
        "positions": len(rows),
        "evals_per_position": round(sum(evals) / len(evals), 1),
        "sampled_expected_loss": mean(exp_loss),
        "sampled_p_loss_over_0.10": mean(blunder),
        "sampled_p_loss_over_0.30": mean(big),
        "mass_on_near_best_moves": mean(near),
        "effective_moves": round(math.exp(sum(ent) / len(ent)), 2),
        "top_move_loss": mean(top_loss),
        "top_move_p_loss_over_0.10": mean(top_blunder),
        "played_loss": mean(played_loss),
        "played_p_loss_over_0.10": mean(played_blunder),
        "played_target_share": mean(played_share),
        "cpu_seconds_per_position": round(sum(secs) / len(secs), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dump", type=Path, help="write every position's target and move (JSONL)")
    a = ap.parse_args()

    positions = json.loads(a.positions.read_text())
    if a.limit:
        positions = positions[: a.limit]
    jobs = [(i, p["moves"], a.configs, a.seed) for i, p in enumerate(positions)]
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(a.ckpt),)) as pool:
        for rows in pool.map(work, jobs, chunksize=4):
            results.extend(rows)
    wall = time.time() - t0

    step = int(a.ckpt.stem.split("_")[1]) if a.ckpt.stem.startswith("step_") else None
    print(
        f"{a.ckpt} (step {step}), {len(positions)} positions, {wall:.0f} s on {a.workers} workers"
    )
    print(
        f"{'setting':<15} {'evals':>6} {'sampled loss':>12} {'p>0.10':>7} {'p>0.30':>7} "
        f"{'top loss':>8} {'played loss':>11} {'eff moves':>9}"
    )
    lines = []
    for config in a.configs:
        record = {"checkpoint": str(a.ckpt), "step": step, "setting": config}
        for phase in ("all", "opening", "later"):
            idx = {i for i, p in enumerate(positions) if phase == "all" or p["phase"] == phase}
            rows = sorted(
                (r for r in results if r["config"] == config and r["index"] in idx),
                key=lambda r: r["index"],
            )
            record[phase] = summarize(rows, [positions[r["index"]]["scores"] for r in rows])
        s = record["all"]
        print(
            f"{config:<15} {s['evals_per_position']:>6.0f} {s['sampled_expected_loss']:>12.4f} "
            f"{s['sampled_p_loss_over_0.10']:>7.3f} {s['sampled_p_loss_over_0.30']:>7.3f} "
            f"{s['top_move_loss']:>8.4f} {s['played_loss']:>11.4f} {s['effective_moves']:>9.2f}"
        )
        lines.append(json.dumps(record))
    if a.out:
        with a.out.open("a") as f:
            f.write("\n".join(lines) + "\n")
    if a.dump:
        with a.dump.open("a") as f:
            for r in sorted(results, key=lambda r: (r["index"], r["config"])):
                row = {"step": step, "phase": positions[r["index"]]["phase"], **r}
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
