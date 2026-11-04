"""INT8 training (model.int8): the forward pass rounds as the fused self-play kernel's INT8 path does
(src/export/int8.py is the specification), and gradients pass straight through the rounding (STE).

Weights: symmetric, one scale per output row, max |w| / 127. Activations: symmetric, one scale per token (per row
of the last dimension), max |a| / 127, round half to even, from the FP32 values (the norms' outputs: the kernel
quantizes inside the norm) or from the FP16-rounded ones (fp16: the attention output and the SwiGLU output, which
the kernel quantizes from FP16 rows), in `groups` equal parts (the down projection's two hidden halves). The scales
are not trained: they follow the values (no gradient through the max).
"""

from __future__ import annotations

import torch

QMAX = 127.0


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
