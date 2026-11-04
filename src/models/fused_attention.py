"""Fused attention for the encoder's training step (CUDA, FP16).

One Triton program per (position, head) runs the QK-norm, the scores with the learned pair bias,
the softmax and the value product, reading q, k and v straight from the qkv projection's output and
writing only the head's 16 output columns. The backward recomputes the scores in one program per
(head, 64 positions) and writes the qkv gradient directly, summing the pair bias's and the
norm weights' gradients over its positions before adding them in. PyTorch's version (two
RMSNorms, the layout copies, scaled_dot_product_attention) moved the [batch, 8, 64, 16] q, k, v and
their gradients through memory several times; the training step is bound by that traffic.

Same function as Block.forward's attention: q and k normed in FP32 then rounded to FP16, scores
scaled by 1 / sqrt(16), the bias added in FP32 (PyTorch's path rounds it to FP16 first).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

TOKENS, HEADS, HEAD_DIM = 64, 8, 16
D_MODEL = HEADS * HEAD_DIM
# Positions per backward program, which sums the pair bias's and norm weights' gradients over
# them before one atomic add: at batch 2,048 on a 4090 the backward takes 1,495 us with 1, 584 with
# 16, 375 with 64 and 511 with 128 (too few programs left).
_BLOCK = 64


@triton.jit
def _fwd_kernel(qkv, qw, kw, bias, out, eps, T: tl.constexpr, H: tl.constexpr, HD: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    D: tl.constexpr = H * HD
    rows = tl.arange(0, T)
    cols = tl.arange(0, HD)
    src = qkv + b * T * 3 * D + rows[:, None] * (3 * D) + h * HD + cols[None, :]
    q = tl.load(src).to(tl.float32)
    k = tl.load(src + D).to(tl.float32)
    v = tl.load(src + 2 * D)
    rq = tl.rsqrt(tl.sum(q * q, 1) / HD + eps)
    rk = tl.rsqrt(tl.sum(k * k, 1) / HD + eps)
    qn = (q * rq[:, None] * tl.load(qw + cols)[None, :]).to(tl.float16)
    kn = (k * rk[:, None] * tl.load(kw + cols)[None, :]).to(tl.float16)
    s = tl.dot(qn, tl.trans(kn)) * 0.25 + tl.load(bias + h * T * T + rows[:, None] * T + rows[None, :])
    p = tl.exp(s - tl.max(s, 1)[:, None])
    p = p / tl.sum(p, 1)[:, None]
    o = tl.dot(p.to(tl.float16), v)
    tl.store(out + b * T * D + rows[:, None] * D + h * HD + cols[None, :], o.to(tl.float16))


@triton.jit
def _bwd_kernel(qkv, qw, kw, bias, out, dout, dqkv, dqw, dkw, dbias, eps, n_pos,
                T: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    c = tl.program_id(0)
    h = tl.program_id(1)
    D: tl.constexpr = H * HD
    rows = tl.arange(0, T)
    cols = tl.arange(0, HD)
    wq = tl.load(qw + cols)
    wk = tl.load(kw + cols)
    bias_h = tl.load(bias + h * T * T + rows[:, None] * T + rows[None, :])
    acc_db = tl.zeros((T, T), dtype=tl.float32)
    acc_dqw = tl.zeros((HD,), dtype=tl.float32)
    acc_dkw = tl.zeros((HD,), dtype=tl.float32)
    for i in range(BLOCK):
        b = c * BLOCK + i
        if b < n_pos:
            off = b * T * 3 * D + rows[:, None] * (3 * D) + h * HD + cols[None, :]
            q = tl.load(qkv + off).to(tl.float32)
            k = tl.load(qkv + off + D).to(tl.float32)
            v = tl.load(qkv + off + 2 * D)
            ooff = b * T * D + rows[:, None] * D + h * HD + cols[None, :]
            o = tl.load(out + ooff).to(tl.float32)
            do = tl.load(dout + ooff)
            rq = tl.rsqrt(tl.sum(q * q, 1) / HD + eps)
            rk = tl.rsqrt(tl.sum(k * k, 1) / HD + eps)
            qn = (q * rq[:, None] * wq[None, :]).to(tl.float16)
            kn = (k * rk[:, None] * wk[None, :]).to(tl.float16)
            s = tl.dot(qn, tl.trans(kn)) * 0.25 + bias_h
            p = tl.exp(s - tl.max(s, 1)[:, None])
            p = p / tl.sum(p, 1)[:, None]
            dv = tl.dot(tl.trans(p.to(tl.float16)), do)
            dp = tl.dot(do, tl.trans(v))
            delta = tl.sum(do.to(tl.float32) * o, 1)
            ds = p * (dp - delta[:, None])
            acc_db += ds
            ds16 = ds.to(tl.float16)
            dqn = tl.dot(ds16, kn) * 0.25
            dkn = tl.dot(tl.trans(ds16), qn) * 0.25
            # RMSNorm backward: y = x r w, r = (mean(x^2) + eps)^-1/2, so with g = dy w,
            # dx = r g - x r^3 mean(x g), and dw = sum over rows of dy x r.
            acc_dqw += tl.sum(dqn * q * rq[:, None], 0)
            acc_dkw += tl.sum(dkn * k * rk[:, None], 0)
            gq = dqn * wq[None, :]
            gk = dkn * wk[None, :]
            dq = rq[:, None] * gq - q * (rq * rq * rq)[:, None] * (tl.sum(q * gq, 1) / HD)[:, None]
            dk = rk[:, None] * gk - k * (rk * rk * rk)[:, None] * (tl.sum(k * gk, 1) / HD)[:, None]
            tl.store(dqkv + off, dq.to(tl.float16))
            tl.store(dqkv + off + D, dk.to(tl.float16))
            tl.store(dqkv + off + 2 * D, dv.to(tl.float16))
    tl.atomic_add(dbias + h * T * T + rows[:, None] * T + rows[None, :], acc_db)
    tl.atomic_add(dqw + cols, acc_dqw)
    tl.atomic_add(dkw + cols, acc_dkw)


@torch.library.custom_op("chessrl::fused_attention", mutates_args=())
def fused_attention(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """qkv [B, 64, 384] FP16 (q | k | v, head-major in each), the norms' weights [16], the pair
    bias [8, 64, 64] FP32 -> the attention output [B, 64, 128] FP16."""
    qkv = qkv.contiguous()
    out = torch.empty(qkv.shape[0], TOKENS, D_MODEL, dtype=torch.float16, device=qkv.device)
    _fwd_kernel[(qkv.shape[0] * HEADS,)](
        qkv, q_weight.float().contiguous(), k_weight.float().contiguous(), bias.float().contiguous(),
        out, eps, T=TOKENS, H=HEADS, HD=HEAD_DIM, num_warps=4,
    )  # fmt: skip
    return out


@fused_attention.register_fake
def _(qkv, q_weight, k_weight, bias, eps):
    return qkv.new_empty(qkv.shape[0], TOKENS, D_MODEL, dtype=torch.float16)


@torch.library.custom_op("chessrl::fused_attention_backward", mutates_args=())
def fused_attention_backward(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, bias: torch.Tensor,
    out: torch.Tensor, dout: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    qkv, out = qkv.contiguous(), out.contiguous()
    dout = dout.to(torch.float16).contiguous()
    n = qkv.shape[0]
    dqkv = torch.empty_like(qkv, dtype=torch.float16)
    dqw = torch.zeros(HEAD_DIM, dtype=torch.float32, device=qkv.device)
    dkw = torch.zeros(HEAD_DIM, dtype=torch.float32, device=qkv.device)
    dbias = torch.zeros(HEADS, TOKENS, TOKENS, dtype=torch.float32, device=qkv.device)
    _bwd_kernel[(triton.cdiv(n, _BLOCK), HEADS)](
        qkv, q_weight.float().contiguous(), k_weight.float().contiguous(), bias.float().contiguous(),
        out, dout, dqkv, dqw, dkw, dbias, eps, n, T=TOKENS, H=HEADS, HD=HEAD_DIM, BLOCK=_BLOCK,
        num_warps=4,
    )  # fmt: skip
    return dqkv, dqw, dkw, dbias


@fused_attention_backward.register_fake
def _(qkv, q_weight, k_weight, bias, out, dout, eps):
    return (
        torch.empty_like(qkv, dtype=torch.float16),
        q_weight.new_empty(HEAD_DIM, dtype=torch.float32),
        k_weight.new_empty(HEAD_DIM, dtype=torch.float32),
        bias.new_empty(HEADS, TOKENS, TOKENS, dtype=torch.float32),
    )


def _setup_context(ctx, inputs, output):
    qkv, q_weight, k_weight, bias, eps = inputs
    ctx.save_for_backward(qkv, q_weight, k_weight, bias, output)
    ctx.eps = eps


def _backward(ctx, dout):
    qkv, q_weight, k_weight, bias, out = ctx.saved_tensors
    dqkv, dqw, dkw, dbias = fused_attention_backward(qkv, q_weight, k_weight, bias, out, dout, ctx.eps)
    return dqkv.to(qkv.dtype), dqw.to(q_weight.dtype), dkw.to(k_weight.dtype), dbias.to(bias.dtype), None


fused_attention.register_autograd(_backward, setup_context=_setup_context)


def reference_attention(qkv, q_weight, k_weight, bias, eps):
    """The same function in plain PyTorch (FP32 norms, FP16 scores' inputs), for the checks."""
    b = qkv.shape[0]
    q, k, v = qkv.view(b, TOKENS, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)

    def norm(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(torch.float16)

    qn, kn = norm(q, q_weight), norm(k, k_weight)
    s = (qn.float() @ kn.float().transpose(-1, -2)) * 0.25 + bias.float()
    o = s.softmax(-1).to(torch.float16).float() @ v.float()
    return o.to(torch.float16).transpose(1, 2).reshape(b, TOKENS, D_MODEL)
