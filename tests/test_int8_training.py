"""INT8 training (model.int8, src/models/quant.py): the network's forward rounds as the INT8 kernel's specification
(src/export/int8.py's reference) does, gradients pass straight through, and the setting travels with the network.
CPU; the fused training kernels' INT8 variants are checked on the GPU (tests/test_fused_block.py, test_fused_ffn.py).
"""

from __future__ import annotations

import numpy as np
import torch

from src.export.heads import ExportWrapper
from src.export.int8 import Int8Linear, int8_reference
from src.export.to_cnet import export
from src.models.encoder import EncoderModel
from src.models.factory import build_model, normalize_spec
from src.models.growth import grow_encoder
from src.models.quant import fq_tokens, fq_weight
from tests.test_cnet_export import read_cnet


def _models(seed: int = 0):
    torch.manual_seed(seed)
    fp = EncoderModel().eval()
    with torch.no_grad():
        fp.pair_bias.normal_(0, 1.0)
        fp.promo.normal_(0, 0.5)
        for blk in fp.blocks:
            blk.x0_gain.fill_(0.3)
    q = EncoderModel(int8=True).eval()
    q.load_state_dict(fp.state_dict())
    return fp, q


def _planes(n: int = 4) -> torch.Tensor:
    rng = np.random.default_rng(0)
    planes = (rng.random((n, 64, 111)) < 0.1).astype(np.float32)
    planes[:, :, 110] = 0.37
    planes[:, :, 104] = 1.0
    return torch.from_numpy(planes).reshape(n, 8, 8, 111)


def test_fake_quant_rounding():
    w = torch.randn(16, 64)
    wq = fq_weight(w)
    scale = w.abs().amax(1, keepdim=True) / 127
    q = wq / scale
    assert torch.allclose(q, q.round(), atol=1e-3) and q.abs().max() <= 127.001
    a = torch.randn(3, 5, 256)
    aq = fq_tokens(a, groups=2)
    for half in aq.chunk(2, -1):
        s = half.abs().amax(-1, keepdim=True) / 127
        assert torch.allclose(half / s, (half / s).round(), atol=1e-3)


def test_straight_through_gradients():
    w = torch.randn(8, 32, requires_grad=True)
    a = torch.randn(4, 32, requires_grad=True)
    (fq_tokens(a) @ fq_weight(w).t()).sum().backward()
    assert torch.allclose(a.grad, w.detach().t().sum(1).expand(4, 32) * 0 + fq_weight(w).detach().sum(0), atol=1e-5)
    assert torch.allclose(w.grad, fq_tokens(a).detach().sum(0).expand(8, 32), atol=1e-5)


def test_projection_matches_the_specification():
    """One projection: the training forward's rounding against src/export/int8.py's (exact integer sums)."""
    torch.manual_seed(1)
    lin = torch.nn.Linear(256, 128, bias=False)
    a = torch.randn(4, 64, 256) * 2
    for fp16, groups in ((False, 1), (True, 2)):
        spec = Int8Linear(lin, groups=groups, fp16=fp16)(a)
        got = torch.nn.functional.linear(fq_tokens(a, fp16=fp16, groups=groups), fq_weight(lin.weight))
        torch.testing.assert_close(got, spec, atol=1e-4, rtol=1e-5)


def test_forward_tracks_the_kernel_specification():
    """The whole network: the same rounding, so much closer to the specification than the FP32 network is. Not
    equal: a value on a rounding boundary rounds differently from FP32 dequantized products than from exact
    integer sums, and attention spreads that over the position."""
    fp, q = _models()
    x = _planes(16)
    with torch.no_grad():
        ref = ExportWrapper(int8_reference(fp), "search", 0.5)(x)
        got = ExportWrapper(q, "search", 0.5)(x)
        plain = ExportWrapper(fp, "search", 0.5)(x)
    assert (got[0] - ref[0]).abs().mean() < 0.6 * (plain[0] - ref[0]).abs().mean()
    assert (got[1] - ref[1]).abs().mean() < 0.3 * (plain[1] - ref[1]).abs().mean()


def test_training_step_moves_every_weight():
    _, q = _models()
    q.train()
    out = q(_planes(), return_all=True)
    (out[0].logsumexp(-1).sum() + sum(o.float().sum() for o in out[1:])).backward()
    for name in ("embed.weight", "blocks.0.qkv.weight", "blocks.0.out.weight", "blocks.0.up.weight",
                 "blocks.0.down.weight", "pol_q.weight", "pol_k.weight"):  # fmt: skip
        g = dict(q.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name


def test_spec_and_growth():
    assert normalize_spec({"arch": "encoder"})["int8"] is False
    model = build_model({"arch": "encoder", "int8": True})
    assert model.spec()["int8"] is True and normalize_spec(model.spec()) == model.spec()
    model.eval()
    with torch.no_grad():
        for blk in model.blocks:
            blk.x0_gain.fill_(0.3)
        bigger = grow_encoder(model, 6)
        x = _planes(2)
        assert bigger.int8 and all(b.int8 for b in bigger.blocks)
        torch.testing.assert_close(bigger(x), model(x))  # growth stays function-preserving under rounding


def test_export_follows_the_network(tmp_path):
    _, q = _models()
    ckpt = {"model": q.state_dict(), "model_spec": q.spec(), "config": {"search_heads": "outcome"}}
    torch.save(ckpt, tmp_path / "q.pt")
    assert export(tmp_path / "q.pt", tmp_path / "q.cnet").endswith(" int8")
    assert "int8" in read_cnet(tmp_path / "q.cnet")
