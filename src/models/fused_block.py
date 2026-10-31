"""Fused attention block for the encoder's training step (CUDA): the embedding's re-injection, the
first RMSNorm, the qkv projection, the attention (QK-norm, pair bias, softmax, values), the out
projection and the residual add, per position.

Forward: one Triton program per position (64 tokens) loops over the 8 heads, as the self-play
kernel does: each head's q, k and v come straight from the normed input, its output goes straight
into the out projection's accumulator, and only the new residual stream (and the normed input,
which a weight gradient needs) reach memory; qkv and the attention output never do.

Backward: the out projection's input gradient (cuBLAS); one program per (head, 64 positions)
recomputes the head's q, k and v from the normed input and runs the attention backward, writing
the qkv gradient and the attention output, with the pair bias's and the QK-norm weights' gradients
summed over its positions before one atomic add; cuBLAS for the input gradient through qkv and
the weight gradients (FP32 accumulators); one kernel for the norm's backward, the residual and the
embedding's gain.

Same function as Block.forward's attention half under FP16 autocast (fused_attention.py's
conventions for the attention itself: q, k, v rounded to FP16 as the qkv projection's output is).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from src.models.fused_attention import reference_attention

TOKENS, HEADS, HEAD_DIM = 64, 8, 16
D = HEADS * HEAD_DIM
_BLOCK = 64  # positions per attention-backward program
_BM = 64  # tokens per norm-backward program


@triton.jit
def _fwd_kernel(x, x0, gain, w1, wqkv, qw, kw, bias, wout, out, xn_out, eps,
                T: tl.constexpr, H: tl.constexpr, HD: tl.constexpr):  # fmt: skip
    pos = tl.program_id(0)
    DM: tl.constexpr = H * HD
    rows = tl.arange(0, T)
    cols = tl.arange(0, DM)
    hc = tl.arange(0, HD)
    off = pos * T * DM + rows[:, None] * DM + cols[None, :]
    g = tl.load(gain)
    xp = tl.load(x + off) + g * tl.load(x0 + off).to(tl.float32)
    r = tl.rsqrt(tl.sum(xp * xp, 1) / DM + eps)
    xn = (xp * r[:, None] * tl.load(w1 + cols)[None, :]).to(tl.float16)
    tl.store(xn_out + off, xn)
    wq_n = tl.load(qw + hc)
    wk_n = tl.load(kw + hc)
    acc = tl.zeros((T, DM), dtype=tl.float32)
    for h in range(H):
        wq = tl.load(wqkv + (h * HD + hc)[:, None] * DM + cols[None, :])
        wk = tl.load(wqkv + (DM + h * HD + hc)[:, None] * DM + cols[None, :])
        wv = tl.load(wqkv + (2 * DM + h * HD + hc)[:, None] * DM + cols[None, :])
        q = tl.dot(xn, tl.trans(wq)).to(tl.float16).to(tl.float32)
        k = tl.dot(xn, tl.trans(wk)).to(tl.float16).to(tl.float32)
        v = tl.dot(xn, tl.trans(wv)).to(tl.float16)
        rq = tl.rsqrt(tl.sum(q * q, 1) / HD + eps)
        rk = tl.rsqrt(tl.sum(k * k, 1) / HD + eps)
        qn = (q * rq[:, None] * wq_n[None, :]).to(tl.float16)
        kn = (k * rk[:, None] * wk_n[None, :]).to(tl.float16)
        s = tl.dot(qn, tl.trans(kn)) * 0.25 + tl.load(bias + h * T * T + rows[:, None] * T + rows[None, :])
        p = tl.exp(s - tl.max(s, 1)[:, None])
        p = p / tl.sum(p, 1)[:, None]
        o = tl.dot(p.to(tl.float16), v).to(tl.float16)
        wo = tl.load(wout + cols[:, None] * DM + (h * HD + hc)[None, :])
        acc += tl.dot(o, tl.trans(wo))
    y = acc.to(tl.float16).to(tl.float32)
    tl.store(out + off, xp + y)


@triton.jit
def _bwd_attn_kernel(xn, wqkv, qw, kw, bias, d_o, o_out, dqkv, dqw, dkw, dbias, eps, n_pos,
                     T: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    h = tl.program_id(0)  # heads vary fastest: the 8 programs of a block read the same xn from L2
    c = tl.program_id(1)
    DM: tl.constexpr = H * HD
    rows = tl.arange(0, T)
    cols = tl.arange(0, DM)
    hc = tl.arange(0, HD)
    wq = tl.load(wqkv + (h * HD + hc)[:, None] * DM + cols[None, :])
    wk = tl.load(wqkv + (DM + h * HD + hc)[:, None] * DM + cols[None, :])
    wv = tl.load(wqkv + (2 * DM + h * HD + hc)[:, None] * DM + cols[None, :])
    wq_n = tl.load(qw + hc)
    wk_n = tl.load(kw + hc)
    bias_h = tl.load(bias + h * T * T + rows[:, None] * T + rows[None, :])
    acc_db = tl.zeros((T, T), dtype=tl.float32)
    acc_dqw = tl.zeros((HD,), dtype=tl.float32)
    acc_dkw = tl.zeros((HD,), dtype=tl.float32)
    for i in range(BLOCK):
        b = c * BLOCK + i
        if b < n_pos:
            xt = tl.load(xn + b * T * DM + rows[:, None] * DM + cols[None, :])
            q = tl.dot(xt, tl.trans(wq)).to(tl.float16).to(tl.float32)
            k = tl.dot(xt, tl.trans(wk)).to(tl.float16).to(tl.float32)
            v = tl.dot(xt, tl.trans(wv)).to(tl.float16)
            rq = tl.rsqrt(tl.sum(q * q, 1) / HD + eps)
            rk = tl.rsqrt(tl.sum(k * k, 1) / HD + eps)
            qn = (q * rq[:, None] * wq_n[None, :]).to(tl.float16)
            kn = (k * rk[:, None] * wk_n[None, :]).to(tl.float16)
            s = tl.dot(qn, tl.trans(kn)) * 0.25 + bias_h
            p = tl.exp(s - tl.max(s, 1)[:, None])
            p = p / tl.sum(p, 1)[:, None]
            o = tl.dot(p.to(tl.float16), v).to(tl.float16)
            hoff = b * T * DM + rows[:, None] * DM + h * HD + hc[None, :]
            tl.store(o_out + hoff, o)
            do = tl.load(d_o + hoff)
            dv = tl.dot(tl.trans(p.to(tl.float16)), do)
            dp = tl.dot(do, tl.trans(v))
            delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
            ds = p * (dp - delta[:, None])
            acc_db += ds
            ds16 = ds.to(tl.float16)
            dqn = tl.dot(ds16, kn) * 0.25
            dkn = tl.dot(tl.trans(ds16), qn) * 0.25
            acc_dqw += tl.sum(dqn * q * rq[:, None], 0)
            acc_dkw += tl.sum(dkn * k * rk[:, None], 0)
            gq = dqn * wq_n[None, :]
            gk = dkn * wk_n[None, :]
            dq = rq[:, None] * gq - q * (rq * rq * rq)[:, None] * (tl.sum(q * gq, 1) / HD)[:, None]
            dk = rk[:, None] * gk - k * (rk * rk * rk)[:, None] * (tl.sum(k * gk, 1) / HD)[:, None]
            qoff = b * T * 3 * DM + rows[:, None] * (3 * DM) + h * HD + hc[None, :]
            tl.store(dqkv + qoff, dq.to(tl.float16))
            tl.store(dqkv + qoff + DM, dk.to(tl.float16))
            tl.store(dqkv + qoff + 2 * DM, dv.to(tl.float16))
    tl.atomic_add(dbias + h * T * T + rows[:, None] * T + rows[None, :], acc_db)
    tl.atomic_add(dqw + hc, acc_dqw)
    tl.atomic_add(dkw + hc, acc_dkw)


@triton.jit
def _bwd_norm_kernel(x, x0, gain, w1, dout, dxn, dx, dx0, dgain, dw1, eps, n_rows,
                     DM: tl.constexpr, BM: tl.constexpr):  # fmt: skip
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    mask = rows < n_rows
    cols = tl.arange(0, DM)
    off = rows[:, None] * DM + cols[None, :]
    g = tl.load(gain)
    x0t = tl.load(x0 + off, mask=mask[:, None], other=0.0).to(tl.float32)
    xp = tl.load(x + off, mask=mask[:, None], other=0.0) + g * x0t
    w = tl.load(w1 + cols)
    r = tl.rsqrt(tl.sum(xp * xp, 1) / DM + eps)
    dn = tl.load(dxn + off, mask=mask[:, None], other=0.0).to(tl.float32)
    gq = dn * w[None, :]
    dxp = tl.load(dout + off, mask=mask[:, None], other=0.0) + r[:, None] * gq - xp * (r * r * r)[:, None] * (
        tl.sum(xp * gq, 1) / DM
    )[:, None]
    tl.store(dx + off, dxp, mask=mask[:, None])
    tl.store(dx0 + off, (g * dxp).to(tl.float16), mask=mask[:, None])
    tl.atomic_add(dgain, tl.sum(tl.sum(x0t * dxp, 1), 0))
    tl.atomic_add(dw1 + cols, tl.sum(dn * xp * r[:, None], 0))


def _no_fp16_acc_mm(a, b):
    """a @ b with FP32 accumulators whatever torch.backends.cuda.matmul.allow_fp16_accumulation
    says: the weight gradients sum over every token of the batch."""
    flag = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    try:
        return a @ b
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = flag


@torch.library.custom_op("chessrl::fused_attention_block", mutates_args=())
def fused_attention_block(
    x: torch.Tensor, x0: torch.Tensor, gain: torch.Tensor, norm_weight: torch.Tensor, qkv_weight: torch.Tensor,
    q_weight: torch.Tensor, k_weight: torch.Tensor, bias: torch.Tensor, out_weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """x [B, 64, 128] FP32 (the residual stream), x0 [B, 64, 128] FP16 (the embedding), the gain,
    norm1's weight [128], the qkv projection's [384, 128], the QK-norm weights [16], the pair bias
    [8, 64, 64], the out projection's [128, 128] -> (x + gain x0 + attention, the normed input)."""
    x = x.float().contiguous()
    x0 = x0.to(torch.float16).contiguous()
    n = x.shape[0]
    out = torch.empty_like(x)
    xn = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    _fwd_kernel[(n,)](
        x, x0, gain.float().reshape(1), norm_weight.float().contiguous(), qkv_weight.half().contiguous(),
        q_weight.float().contiguous(), k_weight.float().contiguous(), bias.float().contiguous(),
        out_weight.half().contiguous(), out, xn, eps, T=TOKENS, H=HEADS, HD=HEAD_DIM, num_warps=8,
    )  # fmt: skip
    return out, xn


@fused_attention_block.register_fake
def _(x, x0, gain, norm_weight, qkv_weight, q_weight, k_weight, bias, out_weight, eps):
    return torch.empty_like(x, dtype=torch.float32), torch.empty_like(x, dtype=torch.float16)


@torch.library.custom_op("chessrl::fused_attention_block_backward", mutates_args=())
def fused_attention_block_backward(
    x: torch.Tensor, x0: torch.Tensor, gain: torch.Tensor, norm_weight: torch.Tensor, qkv_weight: torch.Tensor,
    q_weight: torch.Tensor, k_weight: torch.Tensor, bias: torch.Tensor, out_weight: torch.Tensor,
    xn: torch.Tensor, dout: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:  # fmt: skip
    x = x.float().contiguous()
    x0 = x0.to(torch.float16).contiguous()
    dout = dout.float().contiguous()
    n = x.shape[0]
    dev = x.device
    wqkv16 = qkv_weight.half().contiguous()
    wout16 = out_weight.half().contiguous()
    dy = dout.to(torch.float16)
    d_o = (dy.view(-1, D) @ wout16).view(n, TOKENS, D)  # gradient of the attention output
    o = torch.empty(n, TOKENS, D, dtype=torch.float16, device=dev)
    dqkv = torch.empty(n, TOKENS, 3 * D, dtype=torch.float16, device=dev)
    dqw = torch.zeros(HEAD_DIM, dtype=torch.float32, device=dev)
    dkw = torch.zeros(HEAD_DIM, dtype=torch.float32, device=dev)
    dbias = torch.zeros(HEADS, TOKENS, TOKENS, dtype=torch.float32, device=dev)
    _bwd_attn_kernel[(HEADS, triton.cdiv(n, _BLOCK))](
        xn, wqkv16, q_weight.float().contiguous(), k_weight.float().contiguous(), bias.float().contiguous(),
        d_o, o, dqkv, dqw, dkw, dbias, eps, n, T=TOKENS, H=HEADS, HD=HEAD_DIM, BLOCK=_BLOCK, num_warps=4,
    )  # fmt: skip
    dqkv2 = dqkv.view(-1, 3 * D)
    dxn = dqkv2 @ wqkv16  # [tokens, 128]: K = 384, FP16 accumulation allowed
    dwqkv = _no_fp16_acc_mm(dqkv2.t(), xn.view(-1, D)).float()
    dwout = _no_fp16_acc_mm(dy.view(-1, D).t(), o.view(-1, D)).float()
    rows = n * TOKENS
    dx = torch.empty_like(x)
    dx0 = torch.empty(x.shape, dtype=torch.float16, device=dev)
    dgain = torch.zeros(1, dtype=torch.float32, device=dev)
    dw1 = torch.zeros(D, dtype=torch.float32, device=dev)
    _bwd_norm_kernel[(triton.cdiv(rows, _BM),)](
        x, x0, gain.float().reshape(1), norm_weight.float().contiguous(), dout, dxn, dx, dx0, dgain, dw1, eps,
        rows, DM=D, BM=_BM, num_warps=4,
    )  # fmt: skip
    return dx, dx0, dgain.reshape(()), dw1, dwqkv, dqw, dkw, dbias, dwout


@fused_attention_block_backward.register_fake
def _(x, x0, gain, norm_weight, qkv_weight, q_weight, k_weight, bias, out_weight, xn, dout, eps):
    f32 = torch.float32
    return (
        torch.empty_like(x, dtype=f32), torch.empty_like(x0, dtype=torch.float16), gain.new_empty((), dtype=f32),
        norm_weight.new_empty(D, dtype=f32), qkv_weight.new_empty(3 * D, D, dtype=f32),
        q_weight.new_empty(HEAD_DIM, dtype=f32), k_weight.new_empty(HEAD_DIM, dtype=f32),
        bias.new_empty(HEADS, TOKENS, TOKENS, dtype=f32), out_weight.new_empty(D, D, dtype=f32),
    )  # fmt: skip


def _setup_context(ctx, inputs, output):
    x, x0, gain, norm_weight, qkv_weight, q_weight, k_weight, bias, out_weight, eps = inputs
    ctx.save_for_backward(x, x0, gain, norm_weight, qkv_weight, q_weight, k_weight, bias, out_weight, output[1])
    ctx.eps = eps


def _backward(ctx, dout, dxn_unused):
    x, x0, gain, norm_weight, qkv_weight, q_weight, k_weight, bias, out_weight, xn = ctx.saved_tensors
    grads = fused_attention_block_backward(
        x, x0, gain, norm_weight, qkv_weight, q_weight, k_weight, bias, out_weight, xn, dout, ctx.eps
    )
    inputs = (x, x0, gain, norm_weight, qkv_weight, q_weight, k_weight, bias, out_weight)
    return (*(gd.to(t.dtype) for gd, t in zip(grads, inputs, strict=True)), None)


fused_attention_block.register_autograd(_backward, setup_context=_setup_context)


def reference_attention_block(x, x0, gain, norm_weight, qkv_weight, q_weight, k_weight, bias, out_weight, eps):
    """The same function in plain PyTorch, as Block.forward computes it under FP16 autocast."""
    xp = x.float() + gain * x0.float()
    xn = (xp * torch.rsqrt(xp.pow(2).mean(-1, keepdim=True) + eps) * norm_weight).half()
    qkv = xn @ qkv_weight.half().t()
    o = reference_attention(qkv, q_weight, k_weight, bias, eps)
    return xp + (o @ out_weight.half().t()).float()
