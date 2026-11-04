"""The training step's fused attention (src/models/fused_attention.py) against the same function in
plain PyTorch: the output and the gradients of qkv, both norm weights and the pair bias."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _inputs(batch, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = (torch.randn(batch, 64, 384, device="cuda", generator=g) * 2.0).half().requires_grad_()
    qw = (1.0 + 0.2 * torch.randn(16, device="cuda", generator=g)).requires_grad_()
    kw = (1.0 + 0.2 * torch.randn(16, device="cuda", generator=g)).requires_grad_()
    bias = (0.5 * torch.randn(8, 64, 64, device="cuda", generator=g)).requires_grad_()
    return qkv, qw, kw, bias


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("batch", [1, 37, 64])
def test_fused_attention_matches_reference(batch):
    from src.models.fused_attention import fused_attention, reference_attention

    grads = []
    outs = []
    for fn in (fused_attention, reference_attention):
        qkv, qw, kw, bias = _inputs(batch, 0)
        out = fn(qkv, qw, kw, bias, 1e-6)
        weight = torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out)
        (out.float() * weight).sum().backward()
        outs.append(out.detach())
        grads.append([t.grad for t in (qkv, qw, kw, bias)])
    assert _rel(outs[0], outs[1]) < 2e-3
    for name, a, b in zip(("qkv", "q weight", "k weight", "bias"), grads[0], grads[1], strict=True):
        assert _rel(a, b) < 2e-2, name


def test_encoder_block_uses_it_in_training():
    from src.models.encoder import EncoderModel

    torch.manual_seed(0)
    model = EncoderModel().cuda().train()
    x = torch.randn(8, 8, 8, 111, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        fused = model(x, return_all=True)[1].float()
    import src.models.encoder as enc

    enc._FUSED_ATTENTION = False
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            plain = model(x, return_all=True)[1].float()
    finally:
        enc._FUSED_ATTENTION = True
    assert (fused - plain).abs().max().item() < 2e-2
