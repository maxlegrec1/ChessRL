"""The training step's fused feed-forward (src/models/fused_ffn.py) against the same function in
plain PyTorch under FP16: the output and the gradients of the input, the norm's weight and both
projections."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _inputs(rows, seed, stream=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, 128, device="cuda", generator=g) * 3.0).to(stream).requires_grad_()
    wn = (1.0 + 0.2 * torch.randn(128, device="cuda", generator=g)).requires_grad_()
    wu = (torch.randn(512, 128, device="cuda", generator=g) * 0.09).requires_grad_()
    wd = (torch.randn(128, 256, device="cuda", generator=g) * 0.06).requires_grad_()
    return x, wn, wu, wd


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("stream", [torch.float32, torch.float16])
@pytest.mark.parametrize("rows", [64, 100, 4096])
def test_fused_ffn_matches_reference(rows, stream):
    from src.models.fused_ffn import fused_ffn, reference_ffn

    outs, grads = [], []
    for fn in (fused_ffn, reference_ffn):
        x, wn, wu, wd = _inputs(rows, 0, stream)
        out = fn(x, wn, wu, wd, 1e-6)
        weight = torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out)
        (out.float() * weight).sum().backward()
        outs.append(out.detach())
        grads.append([t.grad for t in (x, wn, wu, wd)])
    assert outs[0].dtype == outs[1].dtype == stream
    assert grads[0][0].dtype == stream
    assert _rel(outs[0], outs[1]) < 1e-3
    for name, a, b in zip(("x", "norm weight", "up", "down"), grads[0], grads[1], strict=True):
        assert _rel(a, b) < 2e-2, name


@pytest.mark.parametrize("fp16_residual", [False, True])
def test_encoder_uses_it_in_training(fp16_residual):
    import src.models.encoder as enc
    from src.models.encoder import EncoderModel

    enc.FP16_RESIDUAL = fp16_residual
    torch.manual_seed(0)
    model = EncoderModel().cuda().train()
    x = torch.randn(8, 8, 8, 111, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        fused = model(x, return_all=True)[1].float()
    enc._FUSED_FFN = False
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            plain = model(x, return_all=True)[1].float()
    finally:
        enc._FUSED_FFN = True
        enc.FP16_RESIDUAL = False
    assert (fused - plain).abs().max().item() < 2e-2
