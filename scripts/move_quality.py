"""How good are the moves self-play actually samples in the opening?

On positions from self-play games (plies 0-29, where the played move is sampled in
proportion to the visit counts), Stockfish scores every legal move (MultiPV, WDL as an
expected score for the side to move). Then, for several ways of choosing the move, the
expected loss against Stockfish's best move and the chance of a blunder:
  policy        the raw policy (no search), sampled
  visits50      visit counts of a 50-sim search as self-play runs it (prior temperature
                1.3, virtual-loss width 8, no root noise): what 75% of opening moves use
  visits400n    visit counts of a 400-sim search with root Dirichlet noise: the other 25%
  selfplay      0.75 * visits50 + 0.25 * visits400n, the distribution self-play samples from
  visits400     a 400-sim search without noise
  played        the move that was actually played in the game (one sample per position)
plus the same for always taking the top move of each.

  uv run python scripts/move_quality.py --ckpt X.pt --engine X.trt --pgn games.pgn [--n 300]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.engine import Net  # noqa: E402
from src.rating import ladder  # noqa: E402


def positions(pgn: Path, n: int, max_ply: int, seed: int) -> list[tuple[list[str], str]]:
    """(moves before the position, move played) for plies < max_ply."""
    out = []
    with pgn.open() as fh:
        while (g := chess.pgn.read_game(fh)) is not None:
            moves = [m.uci() for m in g.mainline_moves()]
            for i in range(min(max_ply, len(moves))):
                out.append((moves[:i], moves[i]))
    random.Random(seed).shuffle(out)
    return out[:n]


def board_of(moves: list[str]) -> chess.Board:
    b = chess.Board()
    for u in moves:
        b.push_uci(u)
    return b


def _sf_scores(args: tuple[list[list[str]], int]) -> list[dict[str, float]]:
    chunk, nodes = args
    eng = chess.engine.SimpleEngine.popen_uci(ladder.stockfish_path())
    eng.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})
    out = []
    try:
        for moves in chunk:
            b = board_of(moves)
            infos = eng.analyse(b, chess.engine.Limit(nodes=nodes), multipv=b.legal_moves.count())
            scores = {}
            for info in infos:
                if "pv" in info and "wdl" in info:
                    w = info["wdl"].relative
                    scores[info["pv"][0].uci()] = (w.wins + 0.5 * w.draws) / 1000.0
            out.append(scores)
    finally:
        eng.quit()
    return out


def stockfish(pos: list[list[str]], nodes: int, procs: int) -> list[dict[str, float]]:
    chunks = [pos[i::procs] for i in range(procs)]
    with mp.Pool(procs) as pool:
        parts = pool.map(_sf_scores, [(c, nodes) for c in chunks])
    res = [None] * len(pos)
    for i, part in enumerate(parts):
        for j, s in enumerate(part):
            res[i + j * procs] = s
    return res


def search(engine: str, pos: list[list[str]], sims: int, noise: bool) -> list[dict[str, float]]:
    import _inference_cpp

    res = _inference_cpp.small_mcts_search_batch_with_history(
        histories=pos, fens=[""] * len(pos), simulations=sims, cpuct=0.65, temperature=1.3,
        engine_path=engine, temperature_at_root=0.0, use_dirichlet=noise, dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25, virtual_loss_width=8)
    return [{m: p for m, p in r["policy"] if p > 0} for r in res]


def summarize(dists: list[dict[str, float]], sf: list[dict[str, float]]) -> dict:
    exp_loss, blunder, big, top_loss, top_blunder, ent, on_best = [], [], [], [], [], [], []
    for d, s in zip(dists, sf, strict=True):
        if not s:
            continue
        best = max(s.values())
        tot = sum(p for m, p in d.items() if m in s) or 1.0
        probs = {m: p / tot for m, p in d.items() if m in s}
        loss = {m: best - v for m, v in s.items()}
        exp_loss.append(sum(p * loss[m] for m, p in probs.items()))
        blunder.append(sum(p for m, p in probs.items() if loss[m] > 0.10))
        big.append(sum(p for m, p in probs.items() if loss[m] > 0.30))
        top = max(probs, key=probs.get)
        top_loss.append(loss[top])
        top_blunder.append(float(loss[top] > 0.10))
        ent.append(-sum(p * math.log(p) for p in probs.values() if p > 0))
        on_best.append(sum(p for m, p in probs.items() if loss[m] < 0.02))
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {"positions": len(exp_loss), "sampled_expected_loss": mean(exp_loss),
            "sampled_p_loss_over_0.10": mean(blunder), "sampled_p_loss_over_0.30": mean(big),
            "mass_on_near_best_moves": mean(on_best), "entropy": mean(ent),
            "top_move_expected_loss": mean(top_loss), "top_move_p_loss_over_0.10": mean(top_blunder)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--pgn", type=Path, required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--max-ply", type=int, default=30)
    p.add_argument("--sf-nodes", type=int, default=300_000)
    p.add_argument("--sf-procs", type=int, default=16)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    pos = positions(a.pgn, a.n, a.max_ply, seed=0)
    hist = [h for h, _ in pos]
    net = Net(a.ckpt, threads=4)
    policy = [{m.uci(): pr for m, pr in priors.items()} for priors, _, _ in net.evaluate([board_of(h) for h in hist])]
    v50 = search(a.engine, hist, 50, noise=False)
    v400n = search(a.engine, hist, 400, noise=True)
    v400 = search(a.engine, hist, 400, noise=False)
    mix = []
    for d50, d400 in zip(v50, v400n, strict=True):
        keys = set(d50) | set(d400)
        mix.append({m: 0.75 * d50.get(m, 0.0) + 0.25 * d400.get(m, 0.0) for m in keys})
    played = [{mv: 1.0} for _, mv in pos]
    sf = stockfish(hist, a.sf_nodes, a.sf_procs)

    out = {"checkpoint": str(a.ckpt), "plies": f"0-{a.max_ply - 1}"}
    for name, d in (("policy", policy), ("visits50", v50), ("visits400n", v400n), ("selfplay", mix),
                    ("visits400", v400), ("played", played)):
        out[name] = summarize(d, sf)
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
