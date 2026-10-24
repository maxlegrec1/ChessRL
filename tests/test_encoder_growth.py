import pytest
import torch

from src.models.encoder import EncoderModel
from src.models.growth import grow_encoder


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_encoder_growth_preserves_function(mode):
    torch.manual_seed(0)
    old = EncoderModel(d_model=32, layers=2, heads=4, d_ff=64)
    with torch.no_grad():
        for p in old.parameters():
            p.add_(torch.randn_like(p) * 0.05)  # move away from init so copying bugs show
    getattr(old, mode)()
    new = grow_encoder(old, 4)
    getattr(new, mode)()
    x = torch.rand(3, 8, 8, 111)
    for a, b in zip(old(x, return_all=True), new(x, return_all=True), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    assert new.spec()["layers"] == 4


def test_new_encoder_layers_get_gradient():
    new = grow_encoder(EncoderModel(d_model=32, layers=1, heads=4, d_ff=64), 2)
    out = new(torch.rand(2, 8, 8, 111), return_all=True)
    sum(o.float().pow(2).mean() for o in out).backward()
    assert new.blocks[1].out.weight.grad.abs().sum() > 0
    assert new.blocks[1].down.weight.grad.abs().sum() > 0
