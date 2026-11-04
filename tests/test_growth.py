import pytest

from src.models.factory import build_model, model_from_checkpoint


def test_checkpoint_roundtrip_uses_spec():
    m = build_model({"arch": "encoder", "d_model": 128, "layers": 2, "heads": 8, "d_ff": 256})
    ckpt = {"model": m.state_dict(), "model_spec": m.spec()}
    assert model_from_checkpoint(ckpt).spec() == m.spec()


def test_resnet_checkpoints_are_refused():
    with pytest.raises(ValueError):
        model_from_checkpoint({"model": {}})  # round 1: no model_spec
    with pytest.raises(ValueError):
        build_model({"arch": "resnet", "blocks": 4, "channels": 64})
