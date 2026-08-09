import importlib

import pytest


@pytest.mark.parametrize(
    "mod",
    [
        "src.train.trainer",
        "src.train.dataset",
        "src.train.config",
        "src.self_play.engine_watch",
        "src.self_play.metrics",
        "src.models.small.small_model",
        "src.models.small.fen_encoder",
        "src.models.small.vocab",
        "src.models.leela.model",
        "src.eval.brawls",
        "src.eval.play_move",
        "src.export.to_onnx",
        "src.generate.openings",
    ],
)
def test_import(mod):
    importlib.import_module(mod)
