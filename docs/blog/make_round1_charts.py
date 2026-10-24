"""Data for the native charts of the first ChessRL post (chessrl-alphazero-single-gpu).

    python3 docs/blog/make_round1_charts.py      # writes site/components/charts/chessrl-round-one.json

The first run's experiment files (artifacts/exp/...) are no longer on this machine. The
match results below are copied from the labels printed on the post's original PNG charts
(exact values; the steps of the Stockfish matches are read off the x axis, to ~0.5k).
The training metrics (legal-move mass, top-1 accuracy, castle rate) come from W&B, exported
to data/round1_wandb.json.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "site" / "components" / "charts" / "chessrl-round-one.json"
WANDB = HERE / "data" / "round1_wandb.json"

# Elo chain, instrumented run: each 5k-step checkpoint against the previous one, 50 games,
# Elo gain of the newer one, for 5k->10k ... 250k->255k.
CHAIN_GAINS = [
    0, 7, 14, 7, -7, 14, 28, -14, 35, 21, 35, 70, 147, 210, 123, 7, 49, 100, 42, 0,
    85, 35, 7, -14, 7, 70, 56, 7, -28, -7, 14, 21, 0, 7, 14, 35, -92, 70, 35, 7,
    0, 0, -14, -35, -21, 14, 49, 56, 21, 28,
]

# Autobrawl, instrumented run: score (%) of each checkpoint against Stockfish 16 at
# UCI_Elo 1400, 100 ms per move, 50 games, steps 160k..255k every 5k.
BRAWL_SCORE = [38, 42, 39, 32, 35, 34, 39, 46, 39, 43, 39, 53, 46, 53, 49, 55, 55, 47, 54, 50]

# Desktop run: Elo implied by the score against Stockfish-1400, (step in thousands, Elo, games).
SF_ELO = [
    (5.2, 983, 30), (10, 1193, 30), (15, 888, 10), (20, 1099, 10), (25, 888, 10), (29, 888, 20),
    (33, 1112, 50), (45, 1289, 100), (45.5, 1227, 50), (50.3, 1227, 50), (50.5, 1257, 100),
    (59, 1289, 100), (64.2, 1253, 100), (65, 1249, 100), (76.7, 1340, 100), (80.3, 1311, 100),
    (84.2, 1376, 100), (91, 1344, 100), (97.9, 1292, 100), (100.7, 1435, 50), (102.8, 1379, 50),
    (105.6, 1407, 50), (110.6, 1344, 50), (113.4, 1393, 100), (115.4, 1428, 50), (129.5, 1442, 50),
    (130.9, 1435, 50), (138.1, 1492, 50),
]

# KQR vs K: 6 positions, the model plays both sides with MCTS.
# (step in thousands, positions mated out of 6, average plies to mate when mated)
KQRK = [
    (5, 3, 10), (10, 4, 13), (15, 2, 11), (20, 2, 15), (25, 4, 28), (30, 3, 22), (35, 4, 36),
    (40, 4, 23), (43.4, 4, 28), (45, 4, 12), (55, 3, 23), (65, 3, 11), (75, 5, 13), (85, 4, 23),
    (95, 3, 14), (100, 5, 11), (105, 6, 16), (110, 6, 15),
]


def rolling(points: list[list[float]], k: int) -> list[list[float]]:
    """Centred rolling mean over k points (fewer at the edges)."""
    out = []
    for i in range(len(points)):
        lo, hi = max(0, i - k // 2), min(len(points), i + k // 2 + 1)
        out.append([points[i][0], sum(p[1] for p in points[lo:hi]) / (hi - lo)])
    return out


def main() -> None:
    cum, chain = 0, [[5, 0]]
    for i, g in enumerate(CHAIN_GAINS):
        cum += g
        chain.append([10 + 5 * i, cum])
    assert abs(cum - 1314) <= 3, cum  # the printed gains are rounded; the exact total was 1314
    brawl = [[160 + 5 * i, s] for i, s in enumerate(BRAWL_SCORE)]
    data = {
        "eloChain": [{"name": "Elo relative to step 5k", "slot": 1, "dots": True, "points": chain}],
        "brawl": [
            {"name": "score, each checkpoint (50 games)", "slot": 1, "dim": True, "dots": True, "points": brawl},
            {"name": "average of 3 neighbouring checkpoints", "slot": 1,
             "points": [[x, round(y, 1)] for x, y in rolling(brawl, 3)]},
        ],
        "sfElo": [{"name": "Elo against Stockfish-1400", "slot": 1, "dots": True,
                   "points": [[s, e] for s, e, _ in SF_ELO]}],
        "sfEloGames": {str(s): g for s, _, g in SF_ELO},
        "kqrk": {
            "rate": [{"name": "positions mated (%)", "slot": 3, "dots": True,
                      "points": [[s, round(100 * n / 6)] for s, n, _ in KQRK]}],
            "plies": [{"name": "plies to mate", "slot": 2, "dots": True, "points": [[s, p] for s, _, p in KQRK]}],
        },
    }
    if WANDB.exists():
        w = json.loads(WANDB.read_text())
        k = lambda pts: [[round(s / 1000, 1), v] for s, v in pts]  # noqa: E731
        data["legal"] = {
            "mass": [{"name": "probability on legal moves", "slot": 1,
                      "points": [[x, round(y, 3)] for x, y in rolling(k(w["legal_move_prob"]), 5)]}],
            "top1": [{"name": "top-1 agreement with the search", "slot": 2,
                      "points": [[x, round(y, 3)] for x, y in rolling(k(w["top1"]), 5)]}],
        }
        data["castle"] = [{"name": "castling moves in the buffer (%)", "slot": 1,
                           "points": [[x, round(y, 3)] for x, y in rolling(k(w["castle_rate_pct"]), 5)]}]
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print("wrote", OUT.name, sorted(data))


if __name__ == "__main__":
    main()
