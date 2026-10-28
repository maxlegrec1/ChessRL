"""How much better are the search's training targets than the raw policy? (and the sampling variants)

The policy only learns while the search improves on it, so the gap between the two is a
signal for whether a run still has room to improve. Both are judged by Stockfish on a
fixed set of positions, scored once and cached:

  build    sample positions from self-play PGNs (plies 0-29 "opening", 30+ "later"), score
           every legal move with Stockfish (MultiPV, WDL as an expected score for the side to
           move), save JSON.
  measure  per checkpoint: the raw policy and a 400-sim search at the self-play settings (no
           root noise) on those positions: expected loss when sampling, loss of the top move,
           share of moves losing > 0.10, the policy -> search gap, and the value head and the
           search's root value against Stockfish. --variants also scores ways of sampling the
           quick moves of the opening phase.

  uv run python scripts/policy_gap.py build --pgn a.pgn b.pgn --out positions.json
  uv run python scripts/policy_gap.py measure --positions positions.json --ckpt X.pt [Y.pt ...] --out gap.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from head_ab import build_engine  # noqa: E402
from move_quality import board_of, stockfish, summarize  # noqa: E402

from src.analysis.engine import Net  # noqa: E402

OPENING_PLIES = 30


def build(a) -> None:
    opening, later = [], []
    for pgn in a.pgn:
        with Path(pgn).open() as fh:
            while (g := chess.pgn.read_game(fh)) is not None:
                moves = [m.uci() for m in g.mainline_moves()]
                for i in range(len(moves)):
                    (opening if i < OPENING_PLIES else later).append((moves[:i], Path(pgn).stem))
    rng = random.Random(a.seed)
    rng.shuffle(opening)
    rng.shuffle(later)
    picked = [(h, s, "opening") for h, s in opening[: a.n_opening]] + [
        (h, s, "later") for h, s in later[: a.n_later]
    ]
    scores = stockfish([h for h, _, _ in picked], a.sf_nodes, a.sf_procs)
    rows = [
        {"moves": h, "source": s, "phase": ph, "scores": sc}
        for (h, s, ph), sc in zip(picked, scores, strict=True)
        if sc
    ]
    Path(a.out).write_text(json.dumps(rows) + "\n")
    print(
        f"wrote {len(rows)} positions ({sum(r['phase'] == 'opening' for r in rows)} opening) to {a.out}"
    )


def search_full(engine: str, hist: list[list[str]], sims: int) -> list[dict]:
    import _inference_cpp

    return _inference_cpp.small_mcts_search_batch_with_history(
        histories=hist,
        fens=[""] * len(hist),
        simulations=sims,
        cpuct=0.65,
        temperature=1.3,
        engine_path=engine,
        temperature_at_root=0.0,
        use_dirichlet=False,
        virtual_loss_width=8,
    )


def sharpen(d: dict[str, float], temp: float) -> dict[str, float]:
    w = {m: p ** (1.0 / temp) for m, p in d.items() if p > 0}
    s = sum(w.values()) or 1.0
    return {m: v / s for m, v in w.items()}


def top_p(d: dict[str, float], p: float) -> dict[str, float]:
    keep, acc = {}, 0.0
    for m, v in sorted(d.items(), key=lambda kv: -kv[1]):
        keep[m] = v
        acc += v
        if acc >= p:
            break
    s = sum(keep.values())
    return {m: v / s for m, v in keep.items()}


def top_k(d: dict[str, float], k: int) -> dict[str, float]:
    keep = dict(sorted(d.items(), key=lambda kv: -kv[1])[:k])
    s = sum(keep.values())
    return {m: v / s for m, v in keep.items()}


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(v * math.log(v / max(q.get(m, 0.0), 1e-9)) for m, v in p.items() if v > 0)


def with_effective(s: dict) -> dict:
    s["effective_moves"] = (
        round(math.exp(s["entropy"]), 2) if s.get("entropy") is not None else None
    )
    return s


def measure_one(ckpt: Path, rows: list[dict], work: Path, variants: bool) -> dict:
    engine = build_engine(ckpt.resolve(), "outcome", work)
    hist = [r["moves"] for r in rows]
    sf = [r["scores"] for r in rows]
    net = Net(ckpt, threads=4)
    evals = net.evaluate([board_of(h) for h in hist])
    policy = [{m.uci(): p for m, p in pri.items()} for pri, _, _ in evals]
    head_e = [w + 0.5 * d for _, (w, d, _l), _ in evals]
    res = search_full(engine, hist, 400)
    visits = [{m: p for m, p in r["policy"] if p > 0} for r in res]
    root_e = [r["value"][0] + 0.5 * r["value"][1] for r in res]
    best_e = [max(s.values()) for s in sf]
    out = {
        "checkpoint": str(ckpt),
        "step": int(ckpt.stem.split("_")[1]) if ckpt.stem.startswith("step_") else None,
    }
    for phase in ("all", "opening", "later"):
        idx = [i for i, r in enumerate(rows) if phase == "all" or r["phase"] == phase]
        pol = with_effective(summarize([policy[i] for i in idx], [sf[i] for i in idx]))
        srch = with_effective(summarize([visits[i] for i in idx], [sf[i] for i in idx]))
        out[phase] = {
            "policy": pol,
            "search400": srch,
            "gap_sampled_loss": round(
                pol["sampled_expected_loss"] - srch["sampled_expected_loss"], 4
            ),
            "gap_top_move_loss": round(
                pol["top_move_expected_loss"] - srch["top_move_expected_loss"], 4
            ),
            "kl_search_vs_policy": round(sum(kl(visits[i], policy[i]) for i in idx) / len(idx), 4),
            "value_head_mae": round(sum(abs(head_e[i] - best_e[i]) for i in idx) / len(idx), 4),
            "search_value_mae": round(sum(abs(root_e[i] - best_e[i]) for i in idx) / len(idx), 4),
        }
    if variants:
        idx = [i for i, r in enumerate(rows) if r["phase"] == "opening"]
        v50 = [
            {m: p for m, p in r["policy"] if p > 0}
            for r in search_full(engine, [hist[i] for i in idx], 50)
        ]
        opts = {
            "visits50 (today)": v50,
            "visits50 top move": [{max(d, key=d.get): 1.0} for d in v50],
            "policy T=1.0": [policy[i] for i in idx],
            "policy T=0.7": [sharpen(policy[i], 0.7) for i in idx],
            "policy T=0.5": [sharpen(policy[i], 0.5) for i in idx],
            "policy top-p 0.95": [top_p(policy[i], 0.95) for i in idx],
            "policy top-p 0.90": [top_p(policy[i], 0.90) for i in idx],
            "policy top-3": [top_k(policy[i], 3) for i in idx],
            "policy top move": [{max(policy[i], key=policy[i].get): 1.0} for i in idx],
        }
        out["opening_variants"] = {
            k: with_effective(summarize(v, [sf[i] for i in idx])) for k, v in opts.items()
        }
    return out


def measure(a) -> None:
    rows = json.loads(Path(a.positions).read_text())
    out = Path(a.out)
    for ck in a.ckpt:
        ck = Path(ck)
        work = Path(a.work) / f"{ck.parent.parent.name}_{ck.stem}"
        work.mkdir(parents=True, exist_ok=True)
        res = measure_one(ck, rows, work, a.variants)
        res["label"] = a.label
        with out.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
        g = res["all"]
        print(
            f"{a.label} {ck.stem}: policy loss {g['policy']['sampled_expected_loss']} -> search "
            f"{g['search400']['sampled_expected_loss']} (gap {g['gap_sampled_loss']}), top move gap "
            f"{g['gap_top_move_loss']}, KL {g['kl_search_vs_policy']}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pgn", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n-opening", type=int, default=300)
    b.add_argument("--n-later", type=int, default=300)
    b.add_argument("--sf-nodes", type=int, default=300_000)
    b.add_argument("--sf-procs", type=int, default=16)
    b.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("measure")
    m.add_argument("--positions", required=True)
    m.add_argument("--ckpt", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--label", default="")
    m.add_argument("--work", default="/workspace/policy_gap_engines")
    m.add_argument("--variants", action="store_true")
    a = p.parse_args()
    build(a) if a.cmd == "build" else measure(a)


if __name__ == "__main__":
    main()
