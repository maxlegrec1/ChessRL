import pytest
import torch

from src.models.factory import build_model, model_from_checkpoint
from src.models.growth import grow_resnet
from src.models.small.small_model import SmallModel


def _outputs(model, x):
    return model(x, return_all=True)


def _trained_like(model: SmallModel) -> SmallModel:
    """Randomize BN affine params and running stats so copying bugs show."""
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.copy_(torch.rand(m.num_features, generator=g) + 0.5)
                m.bias.copy_(torch.randn(m.num_features, generator=g) * 0.1)
                m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
    return model


@pytest.mark.parametrize("target", [(6, 64), (4, 96), (8, 128)])
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_growth_preserves_function(target, mode):
    torch.manual_seed(0)
    old = _trained_like(SmallModel(4, 64))
    x = torch.rand(16, 8, 8, 111)
    getattr(old, mode)()
    new = grow_resnet(old, *target)
    getattr(new, mode)()
    for a, b in zip(_outputs(old, x), _outputs(new, x), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    assert new.spec() == {"arch": "resnet", "blocks": target[0], "channels": target[1]}


def test_grown_model_gets_gradient_on_new_params():
    old = SmallModel(4, 64)
    new = grow_resnet(old, 5, 80)
    new.train()
    out = new(torch.rand(8, 8, 8, 111), return_all=True)
    sum(o.float().pow(2).mean() for o in out).backward()
    assert new.blocks[4].bn2.weight.grad.abs().sum() > 0  # new block
    # Zero-initialised readers of the new channels learn from step one.
    assert new.policy_conv.weight.grad[:, 64:].abs().sum() > 0
    assert new.blocks[0].conv1.weight.grad[:64, 64:].abs().sum() > 0
    assert new.blocks[0].conv2.weight.grad[:64, 64:].abs().sum() > 0


def test_checkpoint_roundtrip_uses_spec():
    m = build_model({"arch": "resnet", "blocks": 3, "channels": 32})
    ckpt = {"model": m.state_dict(), "model_spec": m.spec()}
    m2 = model_from_checkpoint(ckpt)
    assert m2.spec() == m.spec()
    legacy = model_from_checkpoint({"model": SmallModel().state_dict()})
    assert legacy.spec()["blocks"] == 4
