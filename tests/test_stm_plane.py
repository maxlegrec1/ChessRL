"""model.stm_plane: input plane 104 is the real side to move ("colour") or always 1 ("constant", every network
trained before the setting). The setting travels with the network: model_spec, Net, and the .cnet entry."""

import chess
import numpy as np
import pytest

from src.analysis.engine import Net, encode
from src.export.to_cnet import entries_for
from src.models.encoder import EncoderModel
from src.models.factory import build_model, normalize_spec, stm_colour
from src.models.small.fen_encoder import PLANE_STM, fen_to_tensor

AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def test_old_specs_are_constant():
    assert normalize_spec({"arch": "encoder", "d_model": 128})["stm_plane"] == "constant"
    assert normalize_spec(None)["stm_plane"] == "constant"
    assert not stm_colour({"arch": "encoder"})
    assert stm_colour({"arch": "encoder", "stm_plane": "colour"})
    with pytest.raises(ValueError):
        normalize_spec({"arch": "encoder", "stm_plane": "color"})


def test_spec_round_trip():
    model = build_model({"arch": "encoder", "stm_plane": "colour"})
    assert model.spec()["stm_plane"] == "colour"
    assert normalize_spec(model.spec()) == model.spec()


def test_fen_to_tensor_plane():
    black = fen_to_tensor(AFTER_E4, stm_colour=True)
    assert (black[..., PLANE_STM] == 0.0).all()
    assert (fen_to_tensor(AFTER_E4)[..., PLANE_STM] == 1.0).all()
    assert (fen_to_tensor(chess.STARTING_FEN, stm_colour=True)[..., PLANE_STM] == 1.0).all()
    np.testing.assert_array_equal(np.delete(black, PLANE_STM, -1), np.delete(fen_to_tensor(AFTER_E4), PLANE_STM, -1))


def test_mirrored_twin_differs_only_in_the_plane():
    # A position and its colour-mirrored twin encode the same in constant mode (tests/test_canonical.py); in
    # colour mode only plane 104 tells them apart.
    game = chess.Board()
    for san in ("e4", "c5", "Nf3", "d6", "d4"):
        game.push_san(san)
    board = chess.Board(game.fen())  # no move stack, as mirror() gives none
    twin = board.mirror()
    assert board.turn == chess.BLACK and twin.turn == chess.WHITE
    a, b = encode(board, stm_colour=True), encode(twin, stm_colour=True)
    assert (a[..., PLANE_STM] == 0.0).all() and (b[..., PLANE_STM] == 1.0).all()
    np.testing.assert_array_equal(np.delete(a, PLANE_STM, -1), np.delete(b, PLANE_STM, -1))
    np.testing.assert_array_equal(encode(board), encode(twin))


def test_net_follows_the_model():
    assert Net.wrap(EncoderModel(stm_plane="colour")).stm_colour
    assert not Net.wrap(EncoderModel()).stm_colour


def test_cnet_entry():
    names = {name for name, _ in entries_for(EncoderModel().eval(), "outcome")}
    assert "stm_colour" not in names
    entries = dict(entries_for(EncoderModel(stm_plane="colour").eval(), "outcome"))
    np.testing.assert_array_equal(entries["stm_colour"], np.array([1], np.int32))
