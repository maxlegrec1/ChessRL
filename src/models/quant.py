"""INT8 training (model.int8): the forward pass rounds as the fused self-play kernel's INT8 path does
(src/export/int8.py is the specification), and gradients pass straight through the rounding (STE).

Weights: symmetric, one scale per output row, max |w| / 127. Activations: symmetric, one scale per token (per row
of the last dimension), max |a| / 127, round half to even, from the FP32 values (the norms' outputs: the kernel
quantizes inside the norm) or from the FP16-rounded ones (fp16: the attention output and the SwiGLU output, which
the kernel quantizes from FP16 rows), in `groups` equal parts (int8_groups: the SwiGLU output per 128 columns; the
attention output whole at d_model 128, per head in the d_model-512 kernel). The scales are not trained: they follow
the values (no gradient through the max).
"""

from __future__ import annotations

import torch

QMAX = 127.0


def int8_groups(d_model: int, heads: int, d_ff: int) -> tuple[int, int]:
    """(attention output groups, SwiGLU output groups) of an INT8 network's out and down projections, as its kernel
    quantizes them: per 128 SwiGLU columns in both kernels (the d_model-128 one multiplies one 128-column half at a
    time, encoder_fused512.cu one chunk); the attention output per token over all heads at d_model 128
    (encoder_kernel.cu quantizes its rows once every head is in), per token and head otherwise (encoder_fused512.cu
    quantizes each head's output as it is produced)."""
    return (1 if d_model == 128 else heads), max(1, d_ff // 128)


def fq_weight(w: torch.Tensor) -> torch.Tensor:
    """A Linear weight [out, in] rounded to INT8 per output row (to_cnet --int8 writes the same integers)."""
    wf = w.float()
    amax = wf.detach().abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    q = torch.clamp(torch.round(wf / scale), -QMAX, QMAX) * scale
    return (wf + (q - wf).detach()).to(w.dtype)


def fq_tokens(a: torch.Tensor, fp16: bool = False, groups: int = 1) -> torch.Tensor:
    """Activations [..., K] rounded to INT8 per row (and per group of K / groups columns); FP32 out."""
    af = a.float()
    if fp16:
        af = af + (af.half().float() - af).detach()
    parts = []
    for p in af.chunk(groups, dim=-1) if groups > 1 else (af,):
        amax = p.detach().abs().amax(dim=-1, keepdim=True)
        inv = torch.where(amax > 0, QMAX / amax, torch.zeros_like(amax))
        q = torch.round(p * inv) * (amax / QMAX)
        parts.append(p + (q - p).detach())
    return torch.cat(parts, dim=-1) if groups > 1 else parts[0]
