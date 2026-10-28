import importlib

import pytest


@pytest.mark.parametrize(
    "mod",
    [
        "src.run",
        "src.train.trainer",
        "src.train.dataset",
        "src.train.config",
        "src.self_play.engine_watch",
        "src.self_play.calibration",
        "src.models.encoder",
        "src.models.factory",
        "src.models.growth",
        "src.models.small.small_model",
        "src.models.small.fen_encoder",
        "src.models.small.vocab",
        "src.export.to_onnx",
        "src.rating.ladder",
        "src.analysis.engine",
    ],
)
def test_import(mod):
    importlib.import_module(mod)
