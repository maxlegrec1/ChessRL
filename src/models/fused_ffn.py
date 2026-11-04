"""Fused feed-forward for the encoder's training step (CUDA): RMSNorm, the up projection, SwiGLU,
the down projection and the residual add, per tile of tokens, with the hidden layer on chip.

PyTorch's version writes and reads the 512-wide up projection's output (128 MB at batch 2,048),
the 256-wide SwiGLU output and the normed input, forward and backward; the training step is bound
by that traffic. Here the forward reads the residual stream and writes its new value, nothing
else. The backward recomputes the hidden layer in the same tiles, takes the gradient back to the
residual stream and writes what the weight gradients need (the hidden layer's gradient, the SwiGLU
output, the normed input, the output gradient in FP16), which two cuBLAS GEMMs then sum over the
batch with FP32 accumulators (FP16 ones lose 0.3 to 0.5% over its 131,072 tokens).

Same function as Block.forward's feed-forward under FP16 autocast: the norm in FP32, rounded to
FP16; the projections in FP16 with FP32 accumulation, rounded to FP16; SwiGLU rounded to FP16 as
F.silu(gate) * up computes it. The residual stream comes in FP32 or FP16 and leaves in the same
type, its gradient too; the arithmetic is FP32 either way.

quant (INT8 training, src/models/quant.py; the weights come in already rounded): the normed input is rounded to
INT8 per token, and so is the SwiGLU output, per token and hidden half (the self-play kernel multiplies one half
at a time). The forward collects each half from its chunks (one-hot products, exact) before the down projection
and returns each token's max |SwiGLU output| per half, from which the backward rounds its recomputed chunks the
same way for the down projection's weight gradient. Gradients pass straight through the rounding.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from src.models.fused_block import _int8_round, _stream, _stream_dtype
from src.models.quant import fq_tokens

D_MODEL, D_FF = 128, 256
# Tokens per program and hidden units per inner step. At 131,072 tokens on a 4090: the forward takes
# 243 us with 64 tokens (283 with 32), the backward kernel 558 us with 32 (770 with 64, 660 with 64
# on 8 warps); 64 hidden units per step need more shared memory than an SM has. Two pipeline stages
# (217 and 538 us) instead of the default three (245 and 558).
_BM_FWD, _BM_BWD, _BC = 64, 32, 32
# The INT8 forward (quant): 32 tokens per program and each hidden half as one 128-wide chunk (no collecting
# products), one stage: 249 us at 131,072 tokens (458 with the FP16 settings; the FP16 forward 201).
_BM_FWD_Q, _BC_FWD_Q = 32, 128


@triton.jit
def _fwd_kernel(x, wn, wu, wd, out, amax_out, eps, n_rows, D: tl.constexpr, F: tl.constexpr,
                BM: tl.constexpr, BC: tl.constexpr, QUANT: tl.constexpr = False):  # fmt: skip
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < n_rows
    cols = tl.arange(0, D)
    hc = tl.arange(0, BC)
    xt = tl.load(x + rows[:, None] * D + cols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(xt * xt, 1) / D + eps)
    xn32 = xt * r[:, None] * tl.load(wn + cols)[None, :]
    if QUANT:
        xn32 = _int8_round(xn32, tl.max(tl.abs(xn32), 1))
    xn = xn32.to(tl.float16)
    acc = tl.zeros((BM, D), dtype=tl.float32)
    if QUANT:
        HF: tl.constexpr = F // 2
        hcols = tl.arange(0, HF)
        for half in tl.static_range(2):
            gh = tl.zeros((BM, HF), dtype=tl.float32)  # the half's SwiGLU output, chunk by chunk
            for cc in tl.static_range(HF // BC):
                c = half * HF + cc * BC
                wg = tl.load(wu + (c + hc)[:, None] * D + cols[None, :])
                wv = tl.load(wu + (F + c + hc)[:, None] * D + cols[None, :])
                gate = tl.dot(xn, tl.trans(wg)).to(tl.float16).to(tl.float32)
                up = tl.dot(xn, tl.trans(wv)).to(tl.float16).to(tl.float32)
                s = (gate * tl.sigmoid(gate)).to(tl.float16).to(tl.float32)
                g = (s * up).to(tl.float16)
                if BC == HF:  # the half in one chunk
                    gh += g.to(tl.float32)
                else:
                    gh += tl.dot(g, (hcols[None, :] == (cc * BC + hc)[:, None]).to(tl.float16))
            amax = tl.max(tl.abs(gh), 1)
            tl.store(amax_out + rows * 2 + half, amax, mask=mask)
            gq = _int8_round(gh, amax).to(tl.float16)
            wdh = tl.load(wd + cols[:, None] * F + (half * HF + hcols)[None, :])
            acc += tl.dot(gq, tl.trans(wdh))
    else:
        for c in range(0, F, BC):
            wg = tl.load(wu + (c + hc)[:, None] * D + cols[None, :])
            wv = tl.load(wu + (F + c + hc)[:, None] * D + cols[None, :])
            gate = tl.dot(xn, tl.trans(wg)).to(tl.float16).to(tl.float32)
            up = tl.dot(xn, tl.trans(wv)).to(tl.float16).to(tl.float32)
            s = (gate * tl.sigmoid(gate)).to(tl.float16).to(tl.float32)
            g = (s * up).to(tl.float16)
            wdc = tl.load(wd + cols[:, None] * F + (c + hc)[None, :])
            acc += tl.dot(g, tl.trans(wdc))
    y = acc.to(tl.float16).to(tl.float32)
    tl.store(out + rows[:, None] * D + cols[None, :], (xt + y).to(out.dtype.element_ty), mask=mask[:, None])


@triton.jit
def _bwd_kernel(x, wn, wu, wd, dout, dx, du, g_out, xn_out, dy_out, dwn, amax_g, eps, n_rows,
                D: tl.constexpr, F: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
                QUANT: tl.constexpr = False):  # fmt: skip
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < n_rows
    cols = tl.arange(0, D)
    hc = tl.arange(0, BC)
    off = rows[:, None] * D + cols[None, :]
    xt = tl.load(x + off, mask=mask[:, None], other=0.0).to(tl.float32)
    w = tl.load(wn + cols)
    r = tl.rsqrt(tl.sum(xt * xt, 1) / D + eps)
    xn32 = xt * r[:, None] * w[None, :]
    if QUANT:
        xn32 = _int8_round(xn32, tl.max(tl.abs(xn32), 1))
    xn = xn32.to(tl.float16)
    do = tl.load(dout + off, mask=mask[:, None], other=0.0).to(tl.float32)
    dy = do.to(tl.float16)
    tl.store(xn_out + off, xn, mask=mask[:, None])
    tl.store(dy_out + off, dy, mask=mask[:, None])
    acc = tl.zeros((BM, D), dtype=tl.float32)
    for c in range(0, F, BC):
        wg = tl.load(wu + (c + hc)[:, None] * D + cols[None, :])
        wv = tl.load(wu + (F + c + hc)[:, None] * D + cols[None, :])
        wdc = tl.load(wd + cols[:, None] * F + (c + hc)[None, :])
        gate = tl.dot(xn, tl.trans(wg)).to(tl.float16).to(tl.float32)
        up = tl.dot(xn, tl.trans(wv)).to(tl.float16).to(tl.float32)
        sig = tl.sigmoid(gate)
        s = (gate * sig).to(tl.float16).to(tl.float32)
        g = (s * up).to(tl.float16)
        if QUANT:  # the down projection's input as the forward rounded it (its weight gradient reads it)
            a = tl.load(amax_g + rows * 2 + c // (F // 2), mask=mask, other=0.0)
            tl.store(g_out + rows[:, None] * F + (c + hc)[None, :], _int8_round(g.to(tl.float32), a).to(tl.float16),
                     mask=mask[:, None])  # fmt: skip
        else:
            tl.store(g_out + rows[:, None] * F + (c + hc)[None, :], g, mask=mask[:, None])
        dg = tl.dot(dy, wdc)
        dup = (dg * s).to(tl.float16)
        dgate = (dg * up * sig * (1.0 + gate * (1.0 - sig))).to(tl.float16)
        hoff = rows[:, None] * (2 * F) + (c + hc)[None, :]
        tl.store(du + hoff, dgate, mask=mask[:, None])
        tl.store(du + hoff + F, dup, mask=mask[:, None])
        acc += tl.dot(dgate, wg) + tl.dot(dup, wv)
    # RMSNorm backward: y = x r w, r = (mean(x^2) + eps)^-1/2, so with g = dy w,
    # dx = r g - x r^3 mean(x g), and dw = sum over rows of dy x r.
    gq = acc * w[None, :]
    dxn = r[:, None] * gq - xt * (r * r * r)[:, None] * (tl.sum(xt * gq, 1) / D)[:, None]
    tl.store(dx + off, (do + dxn).to(dx.dtype.element_ty), mask=mask[:, None])
    tl.atomic_add(dwn + cols, tl.sum(acc * xt * r[:, None], 0))


@torch.library.custom_op("chessrl::fused_ffn", mutates_args=())
def fused_ffn(
    x: torch.Tensor, norm_weight: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor, eps: float,
    quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., 128] FP32 or FP16 (the residual stream), the norm's weight [128], the up projection's
    [512, 128] (gate rows then up rows) and the down projection's [128, 256] -> (x + FFN(x) in x's
    type, with quant each token's max |SwiGLU output| per hidden half [tokens, 2] (FP32, unset without))."""
    shape = x.shape
    x2 = _stream(x.reshape(-1, D_MODEL))
    n = x2.shape[0]
    out = torch.empty_like(x2)
    amax = torch.empty(n, 2, dtype=torch.float32, device=x2.device)
    bm, bc, stages = (_BM_FWD_Q, _BC_FWD_Q, 1) if quant else (_BM_FWD, _BC, 2)
    _fwd_kernel[(triton.cdiv(n, bm),)](
        x2, norm_weight.float().contiguous(), up_weight.half().contiguous(), down_weight.half().contiguous(),
        out, amax, eps, n, D=D_MODEL, F=D_FF, BM=bm, BC=bc, QUANT=quant, num_warps=4, num_stages=stages,
    )  # fmt: skip
    return out.view(shape), amax


@fused_ffn.register_fake
def _(x, norm_weight, up_weight, down_weight, eps, quant=False):
    return torch.empty_like(x, dtype=_stream_dtype(x)), x.new_empty(x.numel() // D_MODEL, 2, dtype=torch.float32)


@torch.library.custom_op("chessrl::fused_ffn_backward", mutates_args=())
def fused_ffn_backward(
    x: torch.Tensor, norm_weight: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor,
    amax: torch.Tensor, dout: torch.Tensor, eps: float, quant: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    shape = x.shape
    x2 = _stream(x.reshape(-1, D_MODEL))
    d2 = dout.reshape(-1, D_MODEL).to(x2.dtype).contiguous()
    n = x2.shape[0]
    dev = x2.device
    dx = torch.empty_like(x2)
    du = torch.empty(n, 2 * D_FF, dtype=torch.float16, device=dev)
    g = torch.empty(n, D_FF, dtype=torch.float16, device=dev)
    xn = torch.empty(n, D_MODEL, dtype=torch.float16, device=dev)
    dy = torch.empty(n, D_MODEL, dtype=torch.float16, device=dev)
    dwn = torch.zeros(D_MODEL, dtype=torch.float32, device=dev)
    _bwd_kernel[(triton.cdiv(n, _BM_BWD),)](
        x2, norm_weight.float().contiguous(), up_weight.half().contiguous(), down_weight.half().contiguous(),
        d2, dx, du, g, xn, dy, dwn, amax.contiguous(), eps, n, D=D_MODEL, F=D_FF, BM=_BM_BWD, BC=_BC, QUANT=quant,
        num_warps=4, num_stages=2,
    )  # fmt: skip
    # The weight gradients sum over every token: FP32 accumulators whatever the FP16-accumulation
    # setting (torch.backends.cuda.matmul.allow_fp16_accumulation, read when each GEMM is issued).
    fp16_acc = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    try:
        dwu = (du.t() @ xn).float()
        dwd = (dy.t() @ g).float()
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = fp16_acc
    return dx.view(shape), dwn, dwu, dwd


@fused_ffn_backward.register_fake
def _(x, norm_weight, up_weight, down_weight, amax, dout, eps, quant):
    return (
        torch.empty_like(x, dtype=_stream_dtype(x)),
        norm_weight.new_empty(D_MODEL, dtype=torch.float32),
        up_weight.new_empty(2 * D_FF, D_MODEL, dtype=torch.float32),
        down_weight.new_empty(D_MODEL, D_FF, dtype=torch.float32),
    )


def _setup_context(ctx, inputs, output):
    x, norm_weight, up_weight, down_weight, eps, *rest = inputs
    ctx.save_for_backward(x, norm_weight, up_weight, down_weight, output[1])
    ctx.eps = eps
    ctx.quant = bool(rest[0]) if rest else False


def _backward(ctx, dout, damax_unused):
    x, norm_weight, up_weight, down_weight, amax = ctx.saved_tensors
    dx, dwn, dwu, dwd = fused_ffn_backward(x, norm_weight, up_weight, down_weight, amax, dout, ctx.eps, ctx.quant)
    return (
        dx.to(x.dtype), dwn.to(norm_weight.dtype), dwu.to(up_weight.dtype), dwd.to(down_weight.dtype), None, None,
    )  # fmt: skip


fused_ffn.register_autograd(_backward, setup_context=_setup_context)


def reference_ffn(x, norm_weight, up_weight, down_weight, eps, quant=False):
    """The same function in plain PyTorch, as Block.forward computes it under FP16 autocast."""
    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * norm_weight
    if quant:
        xn = fq_tokens(xn)
    gate, up = (xn.half() @ up_weight.half().t()).chunk(2, dim=-1)
    g = torch.nn.functional.silu(gate) * up
    if quant:
        g = fq_tokens(g, fp16=True, groups=2).half()
    return (xf + (g @ down_weight.half().t()).float()).to(_stream_dtype(x))
