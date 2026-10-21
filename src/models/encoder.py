"""Small transformer encoder with the same inputs and five heads as SmallModel.

A scaled-down version of the chess_speedrun recipe that fits the self-play
budget (~1M params): 64 square tokens built from the 111-plane input,
pre-RMSNorm blocks with QK-norm attention, one learned square-pair attention
bias per head shared by all layers ("geom" bias), the input embedding
re-injected before every layer, SwiGLU feed-forward, and an attention policy
head scoring (from-square, to-square) pairs. Attention is plain matmul +
softmax so the legacy ONNX exporter and TensorRT handle it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.small.fen_encoder import INPUT_PLANES
from src.models.small.vocab import policy_index

_PROMO = {"q": 1, "r": 2, "b": 3}


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
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


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
        self.x0_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x, x0, bias):
        b, n, d = x.shape
        x = x + self.x0_gain * x0
        h = self.norm1(x)
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd) + bias
        h = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.out(h)
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

        def mlp(out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, out))

        self.value = mlp(3)
        self.value_q = mlp(3)
        self.mlh = mlp(1)
        self.mlh_q = mlp(1)

    def spec(self) -> dict:
        return {"arch": "encoder", **self.cfg}

    def forward(self, inp, *, return_all: bool = False, **_):
        x = inp[0] if isinstance(inp, (list, tuple)) else inp
        b = x.shape[0]
        x0 = self.embed(x.reshape(b, 64, INPUT_PLANES)) + self.pos
        h = x0
        for block in self.blocks:
            h = block(h, x0, self.pair_bias)
        h = self.norm(h)

        q = self.pol_q(h)  # [B, 64, d]
        k = self.pol_k(h)
        scale = 1.0 / math.sqrt(q.shape[-1])
        q_from = q[:, self.v_from]  # [B, V, d]
        k_to = k[:, self.v_to]
        logits = ((q_from + self.promo[self.v_promo]) * k_to).sum(-1) * scale
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
