"""Fused feed-forward for the encoder's training step (CUDA): RMSNorm, the up projection, SwiGLU,
the down projection and the residual add, per tile of tokens, with the hidden layer on chip.

PyTorch's version writes and reads the 512-wide up projection's output (128 MB at batch 2,048),
the 256-wide SwiGLU output and the normed input, forward and backward; the training step is bound
by that traffic. Here the forward reads the residual stream and writes its new value, nothing
else. The backward recomputes the hidden layer in the same tiles, takes the gradient back to the
residual stream and writes the normed input and the output gradient in FP16. The weight gradients come
from a split-K kernel: one program per (32 hidden units, 2,048 tokens) recomputes its part of the
hidden layer and its gradient from those two and sums its share of both weight gradients with
FP32 accumulators; a sum over the token chunks finishes them. (Writing the hidden layer's gradient
and the SwiGLU output for two cuBLAS GEMMs cost 384 MB of traffic per layer; cuBLAS with FP16
accumulators also loses 0.3 to 0.5% over the batch's 131,072 tokens.)

Same function as Block.forward's feed-forward under FP16 autocast: the norm in FP32, rounded to
FP16; the projections in FP16 with FP32 accumulation, rounded to FP16; SwiGLU rounded to FP16 as
F.silu(gate) * up computes it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

D_MODEL, D_FF = 128, 256
# Tokens per program and hidden units per inner step. At 131,072 tokens on a 4090: the forward takes
# 243 us with 64 tokens (283 with 32), the backward kernel 558 us with 32 (770 with 64, 660 with 64
# on 8 warps); 64 hidden units per step need more shared memory than an SM has.
_BM_FWD, _BM_BWD, _BC = 64, 32, 32
_DW_SLICE, _DW_CHUNK, _DW_STEP = 32, 2048, 64  # split-K weight gradients: hidden units, tokens, inner step


@triton.jit
def _fwd_kernel(x, wn, wu, wd, out, eps, n_rows, D: tl.constexpr, F: tl.constexpr,
                BM: tl.constexpr, BC: tl.constexpr):  # fmt: skip
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < n_rows
    cols = tl.arange(0, D)
    hc = tl.arange(0, BC)
    xt = tl.load(x + rows[:, None] * D + cols[None, :], mask=mask[:, None], other=0.0)
    r = tl.rsqrt(tl.sum(xt * xt, 1) / D + eps)
    xn = (xt * r[:, None] * tl.load(wn + cols)[None, :]).to(tl.float16)
    acc = tl.zeros((BM, D), dtype=tl.float32)
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
    tl.store(out + rows[:, None] * D + cols[None, :], xt + y, mask=mask[:, None])


@triton.jit
def _bwd_kernel(x, wn, wu, wd, dout, dx, xn_out, dy_out, dwn, eps, n_rows,
                D: tl.constexpr, F: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):  # fmt: skip
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < n_rows
    cols = tl.arange(0, D)
    hc = tl.arange(0, BC)
    off = rows[:, None] * D + cols[None, :]
    xt = tl.load(x + off, mask=mask[:, None], other=0.0)
    w = tl.load(wn + cols)
    r = tl.rsqrt(tl.sum(xt * xt, 1) / D + eps)
    xn = (xt * r[:, None] * w[None, :]).to(tl.float16)
    do = tl.load(dout + off, mask=mask[:, None], other=0.0)
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
        dg = tl.dot(dy, wdc)
        dup = (dg * s).to(tl.float16)
        dgate = (dg * up * sig * (1.0 + gate * (1.0 - sig))).to(tl.float16)
        acc += tl.dot(dgate, wg) + tl.dot(dup, wv)
    # RMSNorm backward: y = x r w, r = (mean(x^2) + eps)^-1/2, so with g = dy w,
    # dx = r g - x r^3 mean(x g), and dw = sum over rows of dy x r.
    gq = acc * w[None, :]
    dxn = r[:, None] * gq - xt * (r * r * r)[:, None] * (tl.sum(xt * gq, 1) / D)[:, None]
    tl.store(dx + off, do + dxn, mask=mask[:, None])
    tl.atomic_add(dwn + cols, tl.sum(acc * xt * r[:, None], 0))


@triton.jit
def _dw_kernel(xn, dy, wu, wd, part_up, part_down, n_rows, D: tl.constexpr, F: tl.constexpr,
               SLICE: tl.constexpr, CHUNK: tl.constexpr, STEP: tl.constexpr):  # fmt: skip
    """Partial weight gradients of hidden units [s * SLICE, (s + 1) * SLICE) over the chunk's
    tokens: part_up[c] rows of those gate and up units, part_down[c] those columns."""
    sl = tl.program_id(0)  # slices vary fastest: a chunk's programs read its xn and dy from L2
    ch = tl.program_id(1)
    cols = tl.arange(0, D)
    hs = sl * SLICE + tl.arange(0, SLICE)
    wg = tl.load(wu + hs[:, None] * D + cols[None, :])
    wv = tl.load(wu + (F + hs)[:, None] * D + cols[None, :])
    wdc = tl.load(wd + cols[:, None] * F + hs[None, :])
    acc_g = tl.zeros((SLICE, D), dtype=tl.float32)
    acc_u = tl.zeros((SLICE, D), dtype=tl.float32)
    acc_d = tl.zeros((D, SLICE), dtype=tl.float32)
    for t in range(0, CHUNK, STEP):
        rows = ch * CHUNK + t + tl.arange(0, STEP)
        mask = rows < n_rows
        off = rows[:, None] * D + cols[None, :]
        xt = tl.load(xn + off, mask=mask[:, None], other=0.0)
        dyt = tl.load(dy + off, mask=mask[:, None], other=0.0)
        gate = tl.dot(xt, tl.trans(wg)).to(tl.float16).to(tl.float32)
        up = tl.dot(xt, tl.trans(wv)).to(tl.float16).to(tl.float32)
        sig = tl.sigmoid(gate)
        s = (gate * sig).to(tl.float16).to(tl.float32)
        g = (s * up).to(tl.float16)
        dg = tl.dot(dyt, wdc)
        dup = (dg * s).to(tl.float16)
        dgate = (dg * up * sig * (1.0 + gate * (1.0 - sig))).to(tl.float16)
        acc_g += tl.dot(tl.trans(dgate), xt)
        acc_u += tl.dot(tl.trans(dup), xt)
        acc_d += tl.dot(tl.trans(dyt), g)
    base = ch * (2 * F) * D
    tl.store(part_up + base + hs[:, None] * D + cols[None, :], acc_g)
    tl.store(part_up + base + (F + hs)[:, None] * D + cols[None, :], acc_u)
    tl.store(part_down + ch * D * F + cols[:, None] * F + hs[None, :], acc_d)


@torch.library.custom_op("chessrl::fused_ffn", mutates_args=())
def fused_ffn(
    x: torch.Tensor, norm_weight: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """x [..., 128] FP32 (the residual stream), the norm's weight [128], the up projection's
    [512, 128] (gate rows then up rows) and the down projection's [128, 256] -> x + FFN(x)."""
    shape = x.shape
    x2 = x.reshape(-1, D_MODEL).float().contiguous()
    n = x2.shape[0]
    out = torch.empty_like(x2)
    _fwd_kernel[(triton.cdiv(n, _BM_FWD),)](
        x2, norm_weight.float().contiguous(), up_weight.half().contiguous(), down_weight.half().contiguous(),
        out, eps, n, D=D_MODEL, F=D_FF, BM=_BM_FWD, BC=_BC, num_warps=4,
    )  # fmt: skip
    return out.view(shape)


@fused_ffn.register_fake
def _(x, norm_weight, up_weight, down_weight, eps):
    return torch.empty_like(x, dtype=torch.float32)


@torch.library.custom_op("chessrl::fused_ffn_backward", mutates_args=())
def fused_ffn_backward(
    x: torch.Tensor, norm_weight: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor,
    dout: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    shape = x.shape
    x2 = x.reshape(-1, D_MODEL).float().contiguous()
    d2 = dout.reshape(-1, D_MODEL).float().contiguous()
    n = x2.shape[0]
    dev = x2.device
    dx = torch.empty_like(x2)
    xn = torch.empty(n, D_MODEL, dtype=torch.float16, device=dev)
    dy = torch.empty(n, D_MODEL, dtype=torch.float16, device=dev)
    dwn = torch.zeros(D_MODEL, dtype=torch.float32, device=dev)
    wu16, wd16 = up_weight.half().contiguous(), down_weight.half().contiguous()
    _bwd_kernel[(triton.cdiv(n, _BM_BWD),)](
        x2, norm_weight.float().contiguous(), wu16, wd16,
        d2, dx, xn, dy, dwn, eps, n, D=D_MODEL, F=D_FF, BM=_BM_BWD, BC=_BC, num_warps=4,
    )  # fmt: skip
    chunks = triton.cdiv(n, _DW_CHUNK)
    part_up = torch.empty(chunks, 2 * D_FF, D_MODEL, dtype=torch.float32, device=dev)
    part_down = torch.empty(chunks, D_MODEL, D_FF, dtype=torch.float32, device=dev)
    _dw_kernel[(D_FF // _DW_SLICE, chunks)](
        xn, dy, wu16, wd16, part_up, part_down, n, D=D_MODEL, F=D_FF, SLICE=_DW_SLICE, CHUNK=_DW_CHUNK,
        STEP=_DW_STEP, num_warps=8,
    )  # fmt: skip
    return dx.view(shape), dwn, part_up.sum(0), part_down.sum(0)


@fused_ffn_backward.register_fake
def _(x, norm_weight, up_weight, down_weight, dout, eps):
    return (
        torch.empty_like(x, dtype=torch.float32),
        norm_weight.new_empty(D_MODEL, dtype=torch.float32),
        up_weight.new_empty(2 * D_FF, D_MODEL, dtype=torch.float32),
        down_weight.new_empty(D_MODEL, D_FF, dtype=torch.float32),
    )


def _setup_context(ctx, inputs, output):
    x, norm_weight, up_weight, down_weight, eps = inputs
    ctx.save_for_backward(x, norm_weight, up_weight, down_weight)
    ctx.eps = eps


def _backward(ctx, dout):
    x, norm_weight, up_weight, down_weight = ctx.saved_tensors
    dx, dwn, dwu, dwd = fused_ffn_backward(x, norm_weight, up_weight, down_weight, dout, ctx.eps)
    return (
        dx.to(x.dtype), dwn.to(norm_weight.dtype), dwu.to(up_weight.dtype), dwd.to(down_weight.dtype), None,
    )  # fmt: skip


fused_ffn.register_autograd(_backward, setup_context=_setup_context)


def reference_ffn(x, norm_weight, up_weight, down_weight, eps):
    """The same function in plain PyTorch, as Block.forward computes it under FP16 autocast."""
    xf = x.float()
    xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * norm_weight).half()
    gate, up = (xn @ up_weight.half().t()).chunk(2, dim=-1)
    g = torch.nn.functional.silu(gate) * up
    return xf + (g @ down_weight.half().t()).float()
