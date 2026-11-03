"""INT8 projections for the fused kernel: the weight rounding the exporter writes, and a PyTorch reference of the
kernel's INT8 arithmetic.

Weights: symmetric, one scale per output row (max |w| / 127). Activations: symmetric, one scale per token
(max |a| / 127, q = round-to-nearest-even of a · 127 / max |a|): from the FP32 values for the projections that read
a norm's output (qkv, up, the policy's q and k; the kernel quantizes inside the norm), from the FP16-rounded values
for out and down (the kernel quantizes the attention output and the SwiGLU output rows in shared memory), and per
128-column half for down (the kernel multiplies one hidden half at a time). The input planes are 0/1 except the
halfmove plane (the clock / 100), so the embedding multiplies them as 100 · plane, exactly, at scale 1/100. Integer
products sum exactly in 32 bits; the sum times both scales is the output. Attention scores, softmax, norms, the
residual stream and the value heads keep the FP16/FP32 arithmetic of the FP16 kernel.

int8_reference(model) returns a copy of an encoder whose projections compute exactly that: the forward pass
quantization-aware training should reproduce, and what the INT8 kernel matches (to within roundings the kernel's
other FP16 arithmetic flips). A post-training INT8 export of v3 played the FP16 one to an even score (1,000 games at
200 simulations: Elo +0.3, 95% -14 .. +15).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

QMAX = 127
PLANE_SCALE = 100  # the embedding's inputs: 100 · plane is an integer for every plane (halfmove clock / 100)


def quantize_rows(w: np.ndarray, clip: str = "max") -> tuple[np.ndarray, np.ndarray]:
    """[N, K] float -> (int8 [N, K], FP32 scales [N]) with w ≈ q · scale[:, None]. clip "max": scale = max |w| / 127;
    "mse": the scale in [0.3, 1] · max |w| / 127 with the least squared error per row (large weights clipped)."""
    w = np.asarray(w, np.float32)
    amax = np.abs(w).max(axis=1)
    base = np.where(amax > 0, amax / QMAX, 1.0).astype(np.float32)
    if clip == "max":
        scale = base
    else:
        alphas = np.linspace(0.3, 1.0, 71, dtype=np.float32)
        cand = base[None, :] * alphas[:, None]  # [A, N]
        q = np.clip(np.rint(w[None] / cand[:, :, None]), -QMAX, QMAX)
        err = ((q * cand[:, :, None] - w[None]) ** 2).sum(-1)  # [A, N]
        scale = cand[err.argmin(0), np.arange(w.shape[0])].astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -QMAX, QMAX).astype(np.int8)
    return q, scale


def _quantize_tokens(a: torch.Tensor, fp16: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row of the last dim, as the kernel does it: q = rn(a · 127 / amax), from the FP16 value (fp16: the rows
    quantize_rows makes int8) or the FP32 one (the norms' outputs, norm_to_smem_i8)."""
    a = a.to(torch.float16).float() if fp16 else a.float()
    amax = a.abs().amax(dim=-1, keepdim=True)
    inv = torch.where(amax > 0, QMAX / amax, torch.zeros_like(amax))
    return torch.round(a * inv), amax / QMAX


class Int8Linear(nn.Module):
    """y = Σ q_a q_w · s_a s_w (+ bias); `groups` splits K into equal parts with their own activation scales.
    weights / acts False: that operand stays FP32 (for measuring where the error comes from)."""

    def __init__(self, lin: nn.Linear, groups: int = 1, planes: bool = False, weights: bool = True,
                 acts: bool = True, clip: str = "max", fp16: bool = False):
        super().__init__()
        w = lin.weight.detach().float().cpu().numpy()
        q, s = quantize_rows(w, clip) if weights else (w, np.ones(w.shape[0], np.float32))
        dev = lin.weight.device
        self.register_buffer("qw", torch.from_numpy(q.astype(np.float32)).to(dev))
        self.register_buffer("sw", torch.from_numpy(s).to(dev))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().float().clone())
        self.groups, self.planes, self.acts, self.fp16 = groups, planes, acts, fp16

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        a = a.float()
        if self.planes:
            parts = [(torch.round(a * PLANE_SCALE), torch.full_like(a[..., :1], 1.0 / PLANE_SCALE))]
        elif not self.acts:
            parts = [(c, torch.ones_like(c[..., :1])) for c in a.chunk(self.groups, dim=-1)]
        else:
            parts = [_quantize_tokens(c, self.fp16) for c in a.chunk(self.groups, dim=-1)]
        w_parts = self.qw.chunk(self.groups, dim=-1) if not self.planes else [self.qw]
        y = 0.0
        for (qa, sa), qw in zip(parts, w_parts):
            # Integer sums below 2^24, exact in float64 (no TF32 on this path).
            y = y + (qa.double() @ qw.double().T).float() * sa
        y = y * self.sw
        return y if self.bias is None else y + self.bias.float()


PROJECTIONS = ("embed", "qkv", "out", "up", "down", "pol")
VALUE_MLPS = ("value", "value_q", "mlh", "mlh_q")


@torch.no_grad()
def smooth(model: nn.Module, x: torch.Tensor, alpha: float = 0.5) -> nn.Module:
    """SmoothQuant (Xiao et al., 2023) folded into a copy of the network, the same function in FP32: each projection's
    input channel c is divided by s_c = max|a_c|^alpha / max|w_c|^(1 - alpha) (maxima over the positions x and over
    the weight's rows) and its weight column multiplied by it, which moves activation outliers into the weights. The
    division goes where it costs nothing in the kernel: into the norm weights (qkv, up; the final norm for the policy,
    with the value MLPs' first layers multiplied back), into V's rows (out) and into up's linear half (down)."""
    m = copy.deepcopy(model)
    stats: dict[str, torch.Tensor] = {}
    hooks = []
    named = {"pol_q": m.pol_q}
    for i, b in enumerate(m.blocks):
        named.update({f"{i}.qkv": b.qkv, f"{i}.out": b.out, f"{i}.up": b.up, f"{i}.down": b.down})

    def grab(name):
        def hook(_mod, inp):
            a = inp[0].detach().float().abs().reshape(-1, inp[0].shape[-1]).amax(0)
            stats[name] = torch.maximum(stats[name], a) if name in stats else a
        return hook

    for name, mod in named.items():
        hooks.append(mod.register_forward_pre_hook(grab(name)))
    m(x)
    for h in hooks:
        h.remove()

    def scale(name, *weights):
        wmax = torch.stack([w.abs().amax(0) for w in weights]).amax(0)
        s = stats[name].clamp_min(1e-5) ** alpha / wmax.clamp_min(1e-5) ** (1 - alpha)
        return s / s.log().mean().exp()  # geometric mean 1

    d = m.cfg["d_model"]
    for i, b in enumerate(m.blocks):
        s = scale(f"{i}.qkv", b.qkv.weight)
        b.norm1.weight.div_(s)
        b.qkv.weight.mul_(s)
        s = scale(f"{i}.out", b.out.weight)
        b.qkv.weight[2 * d :].div_(s[:, None])  # V's rows: the attention output channel c is P · V_c
        b.out.weight.mul_(s)
        s = scale(f"{i}.up", b.up.weight)
        b.norm2.weight.div_(s)
        b.up.weight.mul_(s)
        s = scale(f"{i}.down", b.down.weight)
        d_ff = b.down.weight.shape[1]
        b.up.weight[d_ff:].div_(s[:, None])  # silu(gate) · up: the linear half's rows
        b.down.weight.mul_(s)
    s = scale("pol_q", m.pol_q.weight, m.pol_k.weight)
    m.norm.weight.div_(s)
    m.pol_q.weight.mul_(s)
    m.pol_k.weight.mul_(s)
    for name in VALUE_MLPS:  # the pooled vector is the token mean of the final norm's output
        if hasattr(m, name):
            getattr(m, name)[0].weight.mul_(s)
    return m


def int8_reference(model: nn.Module, which=PROJECTIONS, weights: bool = True, acts: bool = True,
                   clip: str = "max") -> nn.Module:
    """A copy of an encoder (src/models/encoder.py) with the fused kernel's INT8 projections (those in `which`)."""
    m = copy.deepcopy(model)
    kw = {"weights": weights, "acts": acts, "clip": clip}
    if "embed" in which:
        m.embed = Int8Linear(m.embed, planes=True, weights=weights, clip=clip)
    for b in m.blocks:
        if "qkv" in which:
            b.qkv = Int8Linear(b.qkv, **kw)
        if "out" in which:
            b.out = Int8Linear(b.out, fp16=True, **kw)
        if "up" in which:
            b.up = Int8Linear(b.up, **kw)
        if "down" in which:
            b.down = Int8Linear(b.down, groups=2, fp16=True, **kw)
    if "pol" in which:
        m.pol_q = Int8Linear(m.pol_q, **kw)
        m.pol_k = Int8Linear(m.pol_k, **kw)
    return m
