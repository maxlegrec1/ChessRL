"""Which value head should the search read? (step 0 of the value-target study)

For one checkpoint, three engines that differ only in the value the search reads
(moves-left stays on the outcome-trained head; see ExportWrapper):
  outcome  - the head trained on game results (what the recipe's self-play used),
  q_value  - the head trained on the search's root WDL,
  blend    - the average of the two.
Then:
  1. ladder: each engine rated against the calibrated anchors, plus puzzle accuracy,
  2. head-to-head: outcome against each of the others on the same random openings,
     both colours,
  3. heads vs Stockfish: on positions from those games, each head's WDL compared
     with Stockfish's own WDL estimate,
  4. Dirichlet bias: the self-play search (400 sims, cpuct 0.65, prior temperature
     1.3, virtual-loss width 8) with root noise on and off, on the same positions:
     how much the noise moves the root value (the value_q training target) and the
     chosen move, and how far the root average sits from the most-visited move's value.

  uv run python scripts/head_ab.py --ckpt step_00061500.pt --out artifacts/head_ab/s61500 --estimate 1890
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.engine import encode  # noqa: E402
from src.models.factory import model_from_checkpoint  # noqa: E402
from src.rating import ladder  # noqa: E402

MODES = ("outcome", "q_value", "blend")


def log(msg: str) -> None:
    print(f"[head_ab {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_engine(ckpt: Path, mode: str, out: Path) -> str:
    onnx, trt = out / f"{mode}.onnx", out / f"{mode}.trt"
    if not trt.exists():
        subprocess.run([sys.executable, "-m", "src.export.to_onnx", "--ckpt_path", str(ckpt),
                        "--onnx_path", str(onnx), "--search_heads", mode], cwd=ROOT, check=True,
                       capture_output=True)
        # Same profile as scripts/build_trt.sh.
        subprocess.run(["trtexec", f"--onnx={onnx}", f"--saveEngine={trt}", "--fp16",
                        "--minShapes=input:1x8x8x111", "--optShapes=input:256x8x8x111",
                        "--maxShapes=input:512x8x8x111", "--shapes=input:256x8x8x111"],
                       check=True, capture_output=True)
    return str(trt)


def elo_diff(score: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400.0 * math.log10(1.0 / score - 1.0)


def summarize_scores(scores: list[float]) -> dict:
    s = np.asarray(scores, dtype=float)
    mean, se = float(s.mean()), float(s.std(ddof=1) / math.sqrt(len(s)))
    lo, hi = elo_diff(mean - 1.96 * se), elo_diff(mean + 1.96 * se)
    return {"games": len(s), "w": int((s == 1).sum()), "d": int((s == 0.5).sum()), "l": int((s == 0).sum()),
            "score": round(mean, 4), "elo": round(elo_diff(mean), 1), "elo_95": [round(lo, 1), round(hi, 1)]}


def rate(engine: str, estimate: float, games_per_anchor: int, sims: int) -> dict:
    ratings = ladder.load_anchor_ratings()
    openings = ladder.make_openings(games_per_anchor // 2)
    results, per_anchor = [], {}
    for name in ladder.pick_anchors(estimate, ratings, k=2):
        anchor = ladder.Anchor(name, seed=7)
        try:
            scores = ladder.play_match(engine, anchor, openings, sims)
        finally:
            anchor.close()
        per_anchor[name] = {"w": scores.count(1.0), "d": scores.count(0.5), "l": scores.count(0.0)}
        results += [(ratings.get(name, 0.0), s) for s in scores]
    elo, se = ladder.fit_rating(results, prior=estimate)
    puzzles = ladder.puzzle_accuracy(engine, ladder.load_puzzles(), sims)
    return {"elo": round(elo, 1), "elo_se": round(se, 1), "anchors": per_anchor, "puzzles": puzzles}


def head_to_head(engine_a: str, engine_b: str, openings: list[list[str]], sims: int,
                 max_plies: int = 300) -> tuple[list[float], list[list[str]]]:
    """A against B, every opening with both colours. Scores from A's side, and the move lists."""
    games = [ladder.Game(o, w) for o in openings for w in (True, False)]  # "model" = A
    while True:
        for g in games:
            if g.result is None:
                over, winner = ladder.rules_end(g.board)
                if over or len(g.history) >= max_plies:
                    g.result = 0.5 if winner is None else (1.0 if winner == g.model_white else 0.0)
        live = [g for g in games if g.result is None]
        if not live:
            break
        # Split before anyone moves: each game makes one move per pass, so the end
        # check above runs before every search (a game A just ended must not reach B).
        batches = [(engine_a, [g for g in live if g.model_to_move]),
                   (engine_b, [g for g in live if not g.model_to_move])]
        for engine, batch in batches:
            if batch:
                for g, uci in zip(batch, ladder.mcts_moves(engine, batch, sims), strict=True):
                    g.board.push_uci(uci)
                    g.history.append(uci)
    return [g.result for g in games], [g.history for g in games]


def sample_positions(histories: list[list[str]], n: int, min_ply: int = 10, seed: int = 0) -> list[list[str]]:
    """Move lists of n positions taken from the games (not terminal, after the random opening)."""
    rng = random.Random(seed)
    cands = []
    for h in histories:
        board = chess.Board()
        for i, uci in enumerate(h):
            board.push_uci(uci)
            if i + 1 >= min_ply and not ladder.rules_end(board)[0]:
                cands.append(h[: i + 1])
    rng.shuffle(cands)
    return cands[:n]


def board_of(moves: list[str]) -> chess.Board:
    b = chess.Board()
    for uci in moves:
        b.push_uci(uci)
    return b


@torch.no_grad()
def head_wdls(ckpt: Path, positions: list[list[str]]) -> dict[str, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_from_checkpoint(torch.load(ckpt, map_location=device, weights_only=False), device).eval()
    out = {"outcome": [], "q_value": []}
    for i in range(0, len(positions), 256):
        x = torch.from_numpy(np.stack([encode(board_of(m)) for m in positions[i:i + 256]])).to(device)
        _, v, vq, _, _ = model(x, return_all=True)
        out["outcome"].append(torch.softmax(v.float(), -1).cpu().numpy())
        out["q_value"].append(torch.softmax(vq.float(), -1).cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["blend"] = 0.5 * (res["outcome"] + res["q_value"])
    return res


def _sf_worker(args: tuple[list[list[str]], int]) -> list[list[float]]:
    chunk, nodes = args
    eng = chess.engine.SimpleEngine.popen_uci(ladder.stockfish_path())
    eng.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})
    out = []
    try:
        for moves in chunk:
            info = eng.analyse(board_of(moves), chess.engine.Limit(nodes=nodes))
            w = info["wdl"].relative  # side to move, per mille
            out.append([w.wins / 1000, w.draws / 1000, w.losses / 1000])
    finally:
        eng.quit()
    return out


def stockfish_wdls(positions: list[list[str]], nodes: int, procs: int) -> np.ndarray:
    chunks = [positions[i::procs] for i in range(procs)]
    with mp.Pool(procs) as pool:
        parts = pool.map(_sf_worker, [(c, nodes) for c in chunks])
    # Re-interleave to the original order.
    res = [None] * len(positions)
    for i, part in enumerate(parts):
        for j, wdl in enumerate(part):
            res[i + j * procs] = wdl
    return np.asarray(res, dtype=float)


def ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def compare_heads(heads: dict[str, np.ndarray], sf: np.ndarray, plies: np.ndarray) -> dict:
    sf_e = sf[:, 0] + 0.5 * sf[:, 1]
    out = {}
    for name, wdl in heads.items():
        e = wdl[:, 0] + 0.5 * wdl[:, 1]
        row = {}
        for label, mask in (("all", plies >= 0), ("ply<40", plies < 40), ("ply40-80", (plies >= 40) & (plies < 80)),
                            ("ply>=80", plies >= 80)):
            if mask.sum() < 20:
                continue
            row[label] = {
                "n": int(mask.sum()),
                "spearman": round(float(np.corrcoef(ranks(e[mask]), ranks(sf_e[mask]))[0, 1]), 4),
                "mae_expected_score": round(float(np.abs(e[mask] - sf_e[mask]).mean()), 4),
                "cross_entropy": round(float(-(sf[mask] * np.log(np.clip(wdl[mask], 1e-6, 1))).sum(1).mean()), 4),
                "argmax_agree": round(float((wdl[mask].argmax(1) == sf[mask].argmax(1)).mean()), 4),
            }
        out[name] = row
    return out


def dirichlet_bias(engine: str, positions: list[list[str]], sims: int) -> dict:
    import _inference_cpp

    def run(noise: bool) -> list[dict]:
        return _inference_cpp.small_mcts_search_batch_with_history(
            histories=positions, fens=[""] * len(positions), simulations=sims, cpuct=0.65, temperature=1.3,
            engine_path=engine, temperature_at_root=0.0, use_dirichlet=noise, dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25, virtual_loss_width=8)

    def stats(res: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        root_e = np.array([r["value"][0] + 0.5 * r["value"][1] for r in res])
        best = [max(r["policy"], key=lambda mv: mv[1])[0] for r in res]
        # q_values: the move's average value from the side to move, in [-1, 1].
        best_e = np.array([(r["q_values"].get(m, 0.0) + 1.0) / 2.0 for r, m in zip(res, best, strict=True)])
        return root_e, best_e, best

    off_root, off_best, off_move = stats(run(False))
    on_root, on_best, on_move = stats(run(True))
    return {
        "positions": len(positions),
        "root_value_shift_mean": round(float((on_root - off_root).mean()), 4),
        "root_value_shift_mean_abs": round(float(np.abs(on_root - off_root).mean()), 4),
        "most_visited_move_changed": round(float(np.mean([a != b for a, b in zip(on_move, off_move, strict=True)])), 4),
        "root_minus_best_noise_off": round(float((off_root - off_best).mean()), 4),
        "root_minus_best_noise_on": round(float((on_root - on_best).mean()), 4),
        "note": "values are expected scores (W + D/2) for the side to move; root = the value_q training target",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--estimate", type=float, required=True, help="current ladder Elo (picks the anchors)")
    p.add_argument("--sims", type=int, default=100, help="search per move in the rating games")
    p.add_argument("--anchor-games", type=int, default=144, help="games per anchor (both colours)")
    p.add_argument("--h2h-openings", type=int, default=200, help="openings per pairing (x2 colours)")
    p.add_argument("--positions", type=int, default=2000)
    p.add_argument("--sf-nodes", type=int, default=1_000_000)
    p.add_argument("--sf-procs", type=int, default=24)
    p.add_argument("--bias-positions", type=int, default=500)
    p.add_argument("--bias-sims", type=int, default=400)
    p.add_argument("--skip-ladder", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    result_path = a.out / "results.json"
    res = json.loads(result_path.read_text()) if result_path.exists() else {}

    def save() -> None:
        result_path.write_text(json.dumps(res, indent=1) + "\n")

    engines = {m: build_engine(a.ckpt.resolve(), m, a.out) for m in MODES}
    log(f"engines built: {list(engines)}")

    if not a.skip_ladder:
        for m in MODES:
            if m not in res.get("ladder", {}):
                log(f"ladder: {m}")
                res.setdefault("ladder", {})[m] = rate(engines[m], a.estimate, a.anchor_games, a.sims)
                save()

    openings = ladder.make_openings(a.h2h_openings, seed=4321)
    histories: list[list[str]] = []
    for other in ("q_value", "blend"):
        key = f"outcome_vs_{other}"
        log(f"head-to-head: {key}")
        scores, hist = head_to_head(engines["outcome"], engines[other], openings, a.sims)
        res.setdefault("head_to_head", {})[key] = summarize_scores(scores)
        histories += hist
        save()

    positions = sample_positions(histories, a.positions)
    plies = np.array([len(m) for m in positions])
    log(f"heads vs Stockfish on {len(positions)} positions")
    heads = head_wdls(a.ckpt, positions)
    sf = stockfish_wdls(positions, a.sf_nodes, a.sf_procs)
    res["heads_vs_stockfish"] = compare_heads(heads, sf, plies)
    np.savez_compressed(a.out / "positions.npz", plies=plies, sf=sf, **heads)
    (a.out / "positions.json").write_text(json.dumps(positions))
    save()

    log("Dirichlet bias")
    res["dirichlet_bias"] = dirichlet_bias(engines["outcome"], positions[: a.bias_positions], a.bias_sims)
    save()
    log(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
