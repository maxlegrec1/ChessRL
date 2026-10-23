"""CPU analysis engine: input encoding, legal-move priors, and the search finding mates."""

import chess
import numpy as np
import pytest
import torch

from src.analysis.engine import Net, analyse, encode, policy_key
from src.models.factory import build_model
from src.models.small.fen_encoder import fen_to_tensor


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return Net.wrap(build_model({"arch": "encoder", "d_model": 32, "layers": 1, "heads": 4, "d_ff": 64}))


def test_encode_without_history_matches_fen_encoder():
    np.testing.assert_array_equal(encode(chess.Board()), fen_to_tensor(chess.STARTING_FEN))


def test_encode_history_and_repetition_planes():
    b = chess.Board()
    for u in ["g1f3", "g8f6", "f3g1", "f6g8"]:  # back to the start position: a repetition
        b.push_uci(u)
    x = encode(b)
    assert x[:, :, 12].all()  # current position seen before
    assert x[:, :, 13:25].any() and x[:, :, 52:64].any()  # t-1 and t-4 piece planes filled
    assert not x[:, :, 64].any()  # t-4 is the start position's first occurrence: no flag yet
    assert not x[:, :, 65:77].any()  # nothing before the first move


def test_priors_are_a_distribution_over_legal_moves(net):
    b = chess.Board("8/4P1k1/8/8/8/8/8/4K3 w - - 0 1")
    (priors, wdl), = net.evaluate([b])
    assert set(priors) == set(b.legal_moves)
    assert sum(priors.values()) == pytest.approx(1.0)
    assert sum(wdl) == pytest.approx(1.0)
    assert policy_key(chess.Move.from_uci("e7e8n")) == "e7e8"  # knight promotion = bare move
    assert policy_key(chess.Move.from_uci("e7e8q")) == "e7e8q"


@pytest.mark.parametrize("fen,mate", [
    ("6k1/5ppp/8/8/8/8/qr3PPP/3R2K1 w - - 0 1", "d1d8"),  # White down a queen and rook, mates
    ("3r2k1/QR3ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "d8d1"),  # the same, for Black
])
def test_search_finds_mate_in_one_with_an_untrained_net(net, fen, mate):
    res = analyse(net, chess.Board(fen), 300)
    assert res["best"] == mate
    assert res["moves"][0]["score"] == pytest.approx(1.0)
