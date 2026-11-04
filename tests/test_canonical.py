"""Canonical positions end to end: the network always plays white. A game and its colour-flipped twin encode to
the same planes, history included; a move and its twin share a vocabulary entry; every legal move of either side
has one; the vocabulary is white's 1,858 moves."""

import random

import chess
import numpy as np
import torch

from src.analysis.engine import encode, policy_key
from src.models.encoder import EncoderModel
from src.models.small.fen_encoder import PLANE_STM
from src.models.small.vocab import mirror_uci, policy_index, policy_to_idx

FENS = [
    chess.STARTING_FEN,
    # Castling rights on one side only, an en passant square, a pawn that can promote by capture or push.
    "r3k2r/pPp2ppp/8/3pP3/8/8/P1PP1PPP/R3K1NR w Kq d6 0 12",
    # Black to move, a black pawn about to promote.
    "8/2P1k3/8/8/8/8/4K1p1/8 b - - 3 40",
]


def _mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move.from_uci(mirror_uci(move.uci()))


def _twin_games(seed: int, plies: int = 40):
    """A random game and its colour-flipped twin, position by position."""
    rng = random.Random(seed)
    game, twin = chess.Board(), chess.Board().mirror()
    for _ in range(plies):
        yield game, twin
        legal = list(game.legal_moves)
        if not legal:
            return
        move = rng.choice(legal)
        game.push(move)
        twin.push(_mirror_move(move))


def test_vocabulary():
    assert len(policy_index) == len(set(policy_index)) == 1858
    assert not any(len(m) == 5 and m[3] == "1" for m in policy_index)  # no black promotions


def test_twin_positions_encode_identically():
    for fen in FENS:
        board = chess.Board(fen)
        x = encode(board)
        np.testing.assert_array_equal(x, encode(board.mirror()))
        assert (x[:, :, PLANE_STM] == 1.0).all()


def test_twin_games_encode_identically_with_history():
    for seed in range(3):
        for game, twin in _twin_games(seed):
            np.testing.assert_array_equal(encode(game), encode(twin), err_msg=game.fen())


def test_every_legal_move_has_an_entry_shared_with_its_twin():
    for seed in range(3):
        for game, twin in _twin_games(seed):
            for move in game.legal_moves:
                key = policy_key(move, game.turn == chess.WHITE)
                assert key in policy_to_idx, (game.fen(), move)
                assert policy_key(_mirror_move(move), twin.turn == chess.WHITE) == key


def test_model_scores_the_vocabulary():
    torch.manual_seed(0)
    x = torch.from_numpy(np.stack([encode(chess.Board(f)) for f in FENS]))
    with torch.no_grad():
        logits = EncoderModel().eval()(x)
    assert logits.shape == (len(FENS), len(policy_index))
    assert torch.isfinite(logits).all()
