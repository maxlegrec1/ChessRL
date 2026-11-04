"""Gumbel root search: the halving schedule, the improved policy, and finding mates."""

import chess
import numpy as np
import pytest
import torch

from src.analysis.engine import Net
from src.analysis.gumbel import considered_visit_sequence, gumbel_search
from src.models.factory import build_model


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return Net.wrap(
        build_model({"arch": "encoder", "d_model": 32, "layers": 1, "heads": 4, "d_ff": 64})
    )


def test_halving_schedule():
    # 4 candidates, 8 simulations: one visit each, then the best 2 get two more each.
    assert considered_visit_sequence(4, 8) == [0, 0, 0, 0, 1, 1, 2, 2]
    assert considered_visit_sequence(1, 5) == [0, 1, 2, 3, 4]
    seq = considered_visit_sequence(16, 32)
    assert len(seq) == 32 and seq[:16] == [0] * 16 and seq[16:] == sorted(seq[16:])
    assert considered_visit_sequence(16, 4) == [0, 0, 0, 0]  # fewer simulations than candidates


@pytest.mark.parametrize("interior", ["gumbel", "puct"])
@pytest.mark.parametrize(
    "fen,mate",
    [
        ("6k1/5ppp/8/8/8/8/qr3PPP/3R2K1 w - - 0 1", "d1d8"),  # White down a queen and rook, mates
        ("3r2k1/QR3ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "d8d1"),  # the same, for Black
    ],
)
def test_finds_mate_in_one_with_an_untrained_net(net, fen, mate, interior):
    board = chess.Board(fen)
    legal = board.legal_moves.count()
    res = gumbel_search(
        net, board, 64, considered=legal, rng=np.random.default_rng(0), interior=interior
    )
    assert res["action"].uci() == mate
    improved = res["improved"]
    assert set(improved) == set(board.legal_moves)
    assert sum(improved.values()) == pytest.approx(1.0)
    assert max(improved, key=improved.get).uci() == mate
    assert res["evals"] <= 1 + 64


def test_same_seed_same_search(net):
    board = chess.Board()
    a = gumbel_search(net, board, 16, rng=np.random.default_rng(7))
    b = gumbel_search(net, board, 16, rng=np.random.default_rng(7))
    assert a["action"] == b["action"] and a["improved"] == b["improved"]
    visited = [c for c in a["root"].children.values() if c.n]
    assert len(visited) == 16 and sum(c.n for c in visited) == 16  # 16 candidates, one visit each
