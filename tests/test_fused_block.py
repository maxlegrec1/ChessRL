"""The training step's fused attention block (src/models/fused_block.py) against the same function
in plain PyTorch under FP16: the output and the gradients of every input."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _inputs(batch, seed, stream=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def t(*shape, scale=1.0, offset=0.0, dtype=torch.float32):
        return (offset + scale * torch.randn(*shape, device="cuda", generator=g)).to(dtype).requires_grad_()

    return (
        t(batch, 64, 128, scale=3.0, dtype=stream),  # x (the residual stream)
        t(batch, 64, 128, dtype=torch.float16),  # x0
        t((), scale=0.3, offset=0.5),  # gain
        t(128, scale=0.2, offset=1.0),  # norm1 weight
        t(384, 128, scale=0.09),  # qkv
        t(16, scale=0.2, offset=1.0),  # q norm
        t(16, scale=0.2, offset=1.0),  # k norm
        t(8, 64, 64, scale=0.5),  # pair bias
        t(128, 128, scale=0.09),  # out
    )


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("stream", [torch.float32, torch.float16])
@pytest.mark.parametrize("batch", [1, 37, 70])
def test_fused_attention_block_matches_reference(batch, stream):
    from src.models.fused_block import fused_attention_block, reference_attention_block

    outs, grads = [], []
    for fused in (True, False):
        args = _inputs(batch, 0, stream)
        if fused:
            out = fused_attention_block(*args, 1e-6)[0]
        else:
            out = reference_attention_block(*args, 1e-6)
        weight = torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out)
        (out.float() * weight).sum().backward()
        outs.append(out.detach())
        grads.append([a.grad for a in args])
    assert outs[0].dtype == outs[1].dtype == stream
    assert grads[0][0].dtype == stream
    assert _rel(outs[0], outs[1]) < 1e-3
    names = ("x", "x0", "gain", "norm1", "qkv", "q norm", "k norm", "bias", "out")
    for name, a, b in zip(names, grads[0], grads[1], strict=True):
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
    enc._FUSED_BLOCK = False
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            plain = model(x, return_all=True)[1].float()
    finally:
        enc._FUSED_BLOCK = True
        enc.FP16_RESIDUAL = False
    assert (fused - plain).abs().max().item() < 2e-2


@pytest.mark.parametrize("batch", [1, 37, 70])
def test_fused_attention_block_int8_matches_reference(batch):
    """INT8 training: the normed input and the attention output (per token, over all heads) rounded, the weights
    rounded per row (fq_weight, outside), gradients straight through; against the same in plain PyTorch."""
    from src.models.fused_block import fused_attention_block, reference_attention_block
    from src.models.quant import fq_weight

    outs, grads = [], []
    for fused in (True, False, None):  # None: the FP16 kernel, to see that the rounding matters
        args = list(_inputs(batch, 0))
        leaves = list(args)
        if fused is None:
            out = fused_attention_block(*args, 1e-6)[0]
        else:
            args[4], args[8] = fq_weight(args[4]), fq_weight(args[8])
            if fused:
                out = fused_attention_block(*args, 1e-6, True)[0]
            else:
                out = reference_attention_block(*args, 1e-6, quant=True)
        weight = torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out)
        (out.float() * weight).sum().backward()
        outs.append(out.detach())
        grads.append([a.grad for a in leaves])
    assert _rel(outs[0], outs[1]) < 1e-3
    assert _rel(outs[0] - outs[2], outs[1] - outs[2]) < 0.12  # the FP16 kernels already differ by ~3e-4
    assert _rel(outs[1], outs[2]) > 1e-3
    names = ("x", "x0", "gain", "norm1", "qkv", "q norm", "k norm", "bias", "out")
    for name, a, b in zip(names, grads[0], grads[1], strict=True):
        assert _rel(a, b) < 3e-2, name


def test_encoder_int8_uses_the_fused_kernels():
    """An int8 network trains through the fused kernels' INT8 variants: the same as its unfused path."""
    import src.models.encoder as enc
    from src.models.encoder import EncoderModel

    torch.manual_seed(0)
    model = EncoderModel(int8=True).cuda().train()
    with torch.no_grad():
        for blk in model.blocks:
            blk.x0_gain.fill_(0.3)
    x = torch.randn(8, 8, 8, 111, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        fused = model(x, return_all=True)
    enc._FUSED_BLOCK = enc._FUSED_FFN = False
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            plain = model(x, return_all=True)
    finally:
        enc._FUSED_BLOCK = enc._FUSED_FFN = True
    assert (fused[1].float() - plain[1].float()).abs().max().item() < 2e-2
    assert _rel(fused[0], plain[0]) < 2e-2
