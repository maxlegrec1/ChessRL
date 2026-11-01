"""Muon on the encoder layers' weight matrices, AdamW on every other parameter (config optimizer: muon).

Muon (Jordan et al. 2024): SGD momentum with Nesterov, then each matrix's update orthogonalised (its singular values
pushed to about 1) before it is applied, with decoupled weight decay. The orthogonalisation is Polar Express (Amsel et
al. 2025, as in the chess speedrun's speedrun/optim.py): five Newton-Schulz-like iterations with one (a, b, c) each.
The fused q, k, v projection is orthogonalised as three 128 x 128 blocks and the up projection as its gate and up
halves; blocks of the same shape across layers are stacked and orthogonalised together. Each update is scaled by
sqrt(max(1, rows / cols)).

The embedding, the pair bias, the norms and gains, and the policy and value heads stay on torch's AdamW (fused on the
GPU). One object drives both so that the trainer, the loss scaler and the checkpoints see a single optimizer: param
groups carry an "lr_scale" (the trainer sets lr = schedule x lr_scale), and with the FP16 loss scaler a step whose
gradients overflowed is skipped on the GPU, without a host sync (found_inf, as fused AdamW does).
"""

from __future__ import annotations

import torch
from torch import nn

# Polar Express coefficients (5 iterations), the matrix first scaled 2% below spectral norm 1.
_POLAR = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def orthogonalize(g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """A batch of matrices [n, rows, cols] with each one's singular values brought to about 1 (FP16 on the GPU:
    after the scaling every entry is below 1; FP32 on the CPU)."""
    x = g.to(torch.float16 if g.is_cuda else torch.float32)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + eps)
    for a, b, c in _POLAR:
        s = x @ x.transpose(-2, -1)
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x.to(g.dtype)


class MuonAdamW:
    """Muon on `matrices` [(param, row blocks)], AdamW on `adamw_groups` (torch param groups)."""

    # The loss scaler hands step() its found_inf (and grad_scale) as attributes, as for fused AdamW.
    _step_supports_amp_scaling = True

    def __init__(
        self,
        matrices: list[tuple[nn.Parameter, int]],
        adamw_groups: list[dict],
        *,
        lr: float,
        betas: tuple[float, float],
        muon_lr: float,
        momentum: float = 0.95,
        muon_weight_decay: float = 0.0,
        fused: bool = False,
    ) -> None:
        self.adamw = torch.optim.AdamW(adamw_groups, lr=lr, betas=betas, fused=fused)
        for group in self.adamw.param_groups:
            group.setdefault("lr_scale", 1.0)
        self.muon_group = {
            "params": [p for p, _ in matrices],
            "blocks": [k for _, k in matrices],
            "lr": muon_lr,
            "lr_scale": muon_lr / lr,
            "momentum": momentum,
            "weight_decay": muon_weight_decay,
        }
        self.param_groups = [*self.adamw.param_groups, self.muon_group]
        self.bufs = [torch.zeros_like(p) for p in self.muon_group["params"]]
        # Same-shape blocks orthogonalised together: shape -> [(param index, block index)].
        self._shapes: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for i, (p, k) in enumerate(matrices):
            if p.ndim != 2 or p.shape[0] % k:
                raise ValueError(f"Muon needs 2-D matrices with {k} row blocks, got {tuple(p.shape)}")
            for j in range(k):
                self._shapes.setdefault((p.shape[0] // k, p.shape[1]), []).append((i, j))

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adamw.zero_grad(set_to_none=set_to_none)
        for p in self.muon_group["params"]:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        found_inf = getattr(self, "found_inf", None)
        grad_scale = getattr(self, "grad_scale", None)
        if self.adamw.defaults.get("fused") and (found_inf is not None or grad_scale is not None):
            self.adamw.grad_scale, self.adamw.found_inf = grad_scale, found_inf
            try:
                self.adamw.step()
            finally:
                del self.adamw.grad_scale, self.adamw.found_inf
        else:
            self.adamw.step()
        self._muon_step(found_inf, grad_scale)

    def _muon_step(self, found_inf: torch.Tensor | None, grad_scale: torch.Tensor | None) -> None:
        group = self.muon_group
        params, blocks = group["params"], group["blocks"]
        if any(p.grad is None for p in params):
            return
        lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
        skip = None if found_inf is None else found_inf > 0
        # Momentum and the Nesterov look-ahead; an overflowed step leaves the buffers as they were.
        upds = []
        for p, buf in zip(params, self.bufs, strict=True):
            g = p.grad.float() if grad_scale is None else p.grad.float() / grad_scale
            new = buf * mom + g
            buf.copy_(new if skip is None else torch.where(skip, buf, new))
            upds.append(g + mom * new)
        # Orthogonalise each shape's blocks as one batch, then write them back into per-parameter updates.
        ortho = [torch.empty_like(u) for u in upds]
        for (rows, cols), members in self._shapes.items():
            stacked = torch.stack([upds[i].view(blocks[i], rows, cols)[j] for i, j in members])
            out = orthogonalize(stacked) * max(1.0, rows / cols) ** 0.5
            for (i, j), o in zip(members, out.unbind(0), strict=True):
                ortho[i].view(blocks[i], rows, cols)[j].copy_(o)
        for p, o in zip(params, ortho, strict=True):
            new = p * (1.0 - lr * wd) - lr * o.to(p.dtype)
            p.copy_(new if skip is None else torch.where(skip, p, new))

    def state_dict(self) -> dict:
        return {"adamw": self.adamw.state_dict(), "muon_bufs": [b.clone() for b in self.bufs]}

    def load_state_dict(self, state: dict) -> None:
        bufs = state["muon_bufs"]
        if len(bufs) != len(self.bufs) or any(a.shape != b.shape for a, b in zip(bufs, self.bufs, strict=True)):
            raise ValueError("Muon buffers do not match the model's matrices")
        self.adamw.load_state_dict(state["adamw"])
        for mine, saved in zip(self.bufs, bufs, strict=True):
            mine.copy_(saved)
