"""Small transformer encoder with the same inputs and five heads as SmallModel.

A scaled-down version of the chess_speedrun recipe that fits the self-play
budget (0.81M parameters at d_model 128, 4 layers): 64 square tokens built from the 111-plane input,
pre-RMSNorm blocks with QK-norm attention, one learned square-pair attention
bias per head shared by all layers ("geom" bias), the input embedding
re-injected before every layer, SwiGLU feed-forward, and an attention policy
head scoring (from-square, to-square) pairs. Attention runs as one fused kernel
(scaled_dot_product_attention, the pair bias as its additive mask); cpp/net/encoder_kernel.cu
runs the same network for self-play.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.small.fen_encoder import INPUT_PLANES
from src.models.small.vocab import policy_index

_PROMO = {"q": 1, "r": 2, "b": 3}

try:  # Triton (with PyTorch's CUDA builds): the fused kernels for training
    from src.models.fused_attention import fused_attention
    from src.models.fused_block import fused_attention_block
    from src.models.fused_ffn import fused_ffn
except ImportError:  # pragma: no cover
    fused_attention = fused_attention_block = fused_ffn = None
# Off switches for A/B runs: the whole attention block (fused_block.py), the attention alone
# (fused_attention.py, used when the block is off), the feed-forward (fused_ffn.py).
_FUSED_BLOCK = os.environ.get("CHESSRL_FUSED_BLOCK", "1") != "0"
_FUSED_ATTENTION = os.environ.get("CHESSRL_FUSED_ATTENTION", "1") != "0"
_FUSED_FFN = os.environ.get("CHESSRL_FUSED_FFN", "1") != "0"
# The residual stream in FP16 under CUDA autocast (the trainer's fp16_residual setting): half the
# bytes a training step moves through it. Off, or outside autocast, it stays FP32. The self-play
# kernel keeps it FP32.
FP16_RESIDUAL = False


def _token(square: int) -> int:
    # Planes put rank 1 at the bottom: row = 7 - rank, so token = (7 - rank) * 8 + file.
    return (7 - square // 8) * 8 + square % 8


def _vocab_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per vocab entry: from-token, to-token, promotion type (0 none, -1 special)."""
    frm, to, promo = [], [], []
    for move in policy_index:
        if len(move) >= 4 and move[0] in "abcdefgh" and move[1].isdigit():
            f = (int(move[1]) - 1) * 8 + "abcdefgh".index(move[0])
            t = (int(move[3]) - 1) * 8 + "abcdefgh".index(move[2])
            frm.append(_token(f))
            to.append(_token(t))
            promo.append(_PROMO.get(move[4:], 0))
        else:
            frm.append(0)
            to.append(0)
            promo.append(-1)
    return torch.tensor(frm), torch.tensor(to), torch.tensor(promo)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # In FP32, as the fused self-play kernel computes it: x.pow(2) cannot overflow there
        # (in FP16 it did past |x| = 256, which an earlier version avoided by dividing each row
        # by its maximum first, at two extra reductions per norm).
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int):
        super().__init__()
        self.heads = heads
        self.hd = d // heads
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)
        self.out = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.up = nn.Linear(d, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
        self.d_ff = d_ff
        self.x0_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x, x0, bias):
        b, n, d = x.shape
        if (
            fused_attention_block is not None
            and _FUSED_BLOCK
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float16)
            and x0.dtype == torch.float16
            and torch.is_autocast_enabled("cuda")
            and torch.is_grad_enabled()
            and (n, self.heads, self.hd) == (64, 8, 16)
        ):
            # Training: the re-injection, norm1, qkv, attention, out projection and residual per
            # position (fused_block.py).
            x, _ = fused_attention_block(
                x, x0, self.x0_gain, self.norm1.weight, self.qkv.weight, self.q_norm.weight,
                self.k_norm.weight, bias, self.out.weight, self.norm1.eps,
            )  # fmt: skip
            return self._feed_forward(x)
        # The re-injection and the attention's residual add in FP32, rounded once to the stream's type
        # (as the fused block computes them).
        xp = x + self.x0_gain * x0.float()
        h = self.norm1(xp)
        qkv = self.qkv(h)
        if (
            fused_attention is not None
            and _FUSED_ATTENTION
            and qkv.is_cuda
            and qkv.dtype == torch.float16
            and torch.is_grad_enabled()
            and (n, self.heads, self.hd) == (64, 8, 16)
        ):
            # Training: QK-norm, scores, softmax and values in one kernel (fused_attention.py).
            h = fused_attention(qkv, self.q_norm.weight, self.k_norm.weight, bias, self.q_norm.eps)
        else:
            q, k, v = qkv.view(b, n, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
            q, k = self.q_norm(q), self.k_norm(k)
            # softmax(q.k / sqrt(hd) + bias) v without writing the [batch, heads, 64, 64] scores:
            # they and their softmax were a quarter of a training step's GPU time. The fused
            # kernel's FP32 backward on the GPU rejects this layout, so FP32 training there keeps
            # the explicit form.
            if q.is_cuda and v.dtype == torch.float32 and torch.is_grad_enabled():
                att = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd) + bias
                h = att.softmax(dim=-1) @ v
            else:
                h = F.scaled_dot_product_attention(
                    q.to(v.dtype), k.to(v.dtype), v, attn_mask=bias.to(v.dtype).expand(b, -1, -1, -1)
                )
            h = h.transpose(1, 2).reshape(b, n, d)
        return self._feed_forward((xp + self.out(h)).to(x.dtype))

    def _feed_forward(self, x):
        d = x.shape[-1]
        if (
            fused_ffn is not None
            and _FUSED_FFN
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float16)
            and torch.is_autocast_enabled("cuda")
            and torch.is_grad_enabled()
            and (d, self.d_ff) == (128, 256)
        ):
            # Training: norm, SwiGLU feed-forward and residual per tile of tokens (fused_ffn.py).
            return fused_ffn(x, self.norm2.weight, self.up.weight, self.down.weight, self.norm2.eps)
        gate, up = self.up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class EncoderModel(nn.Module):
    def __init__(self, d_model: int = 128, layers: int = 4, heads: int = 8, d_ff: int = 256):
        super().__init__()
        self.cfg = {"d_model": d_model, "layers": layers, "heads": heads, "d_ff": d_ff}
        d = d_model
        self.embed = nn.Linear(INPUT_PLANES, d)
        self.pos = nn.Parameter(torch.randn(64, d) * 0.02)
        self.pair_bias = nn.Parameter(torch.zeros(heads, 64, 64))
        self.blocks = nn.ModuleList([Block(d, heads, d_ff) for _ in range(layers)])
        self.norm = RMSNorm(d)
        # Attention policy head.
        self.pol_q = nn.Linear(d, d)
        self.pol_k = nn.Linear(d, d)
        self.promo = nn.Parameter(torch.zeros(4, d))
        self.special_logit = nn.Parameter(torch.full((), -10.0))
        frm, to, promo = _vocab_tables()
        self.register_buffer("v_from", frm, persistent=False)
        self.register_buffer("v_to", to, persistent=False)
        self.register_buffer("v_promo", promo.clamp_min(0), persistent=False)
        self.register_buffer("v_special", promo < 0, persistent=False)
        # Flat indices into the [64 x 64] from/to scores and [4 x 64] promotion scores.
        self.register_buffer("v_ft", frm * 64 + to, persistent=False)
        self.register_buffer("v_pt", promo.clamp_min(0) * 64 + to, persistent=False)

        def mlp(out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, out))

        self.value = mlp(3)
        self.value_q = mlp(3)
        self.mlh = mlp(1)
        self.mlh_q = mlp(1)

    def spec(self) -> dict:
        return {"arch": "encoder", **self.cfg}

    def forward(self, x, *, return_all: bool = False):
        b = x.shape[0]
        x0 = self.embed(x.reshape(b, 64, INPUT_PLANES)) + self.pos
        # The copy re-injected into every layer is FP16 under autocast, as the self-play kernel keeps
        # it: half the bytes of each layer's FP32 read and of its gradient. The residual stream
        # starts from it with FP16_RESIDUAL, from the FP32 embedding otherwise.
        x0_layers = x0.to(torch.float16) if torch.is_autocast_enabled("cuda") and x0.is_cuda else x0
        h = x0_layers if FP16_RESIDUAL else x0
        for block in self.blocks:
            h = block(h, x0_layers, self.pair_bias)
        h = self.norm(h)

        # logit(from, to, promo) = (q_from + promo_vec) . k_to / sqrt(d): two small
        # matmuls, then pick the vocab entries by flat index. (Gathering a d-wide
        # vector per vocab entry first cost ~0.75 GB per TensorRT context.)
        q = self.pol_q(h)  # [B, 64, d]
        k = self.pol_k(h)
        scale = 1.0 / math.sqrt(self.cfg["d_model"])
        kt = k.transpose(1, 2)  # [B, d, 64]
        ft = (q @ kt).reshape(b, 64 * 64)
        pt = (self.promo @ kt).reshape(b, 4 * 64)
        logits = (ft[:, self.v_ft] + pt[:, self.v_pt]) * scale
        logits = torch.where(self.v_special, self.special_logit.to(logits.dtype), logits)
        if not return_all:
            return logits

        pooled = h.mean(dim=1)
        return (
            logits,
            self.value(pooled),
            self.value_q(pooled),
            self.mlh(pooled).squeeze(-1),
            self.mlh_q(pooled).squeeze(-1),
        )
