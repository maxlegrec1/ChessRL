"""The checkpoint's `search_heads` setting: which heads the search reads."""

from __future__ import annotations

import torch

from src.train.config import RampConfig


def search_q_weight(ckpt: dict) -> float:
    """search_heads "blend": the weight of value_q in the WDL the search reads for this checkpoint,
    `search_q_weight`, or `search_q_ramp` at the checkpoint's step when that ramp is on."""
    config = ckpt.get("config") or {}
    ramp = RampConfig(**(config.get("search_q_ramp") or {}))
    if ramp.on:
        return ramp.at(int(ckpt.get("step", 0) or 0))
    return float(config.get("search_q_weight", 0.5))


class ExportWrapper(torch.nn.Module):
    """Picks which heads feed the network's "value" and "mlh_q" outputs, in PyTorch.

    The C++ MCTS reads "value" as the leaf WDL and "mlh_q" as the leaf
    plies-remaining estimate (depth-incremented backup + tiebreaker). The .cnet
    export (to_cnet.HEADS) computes the same outputs in the fused kernel; this is
    the reference the tests and scripts/net_parity.py compare it with.
      search_heads="search":         value_q / mlh_q, trained on MCTS backups.
      search_heads="outcome":        value / mlh, trained on game results and
                                     true plies remaining (AlphaZero-style).
    Value-only variants (moves-left stays on the outcome-trained mlh, so only the
    value the search reads changes):
      search_heads="q_value":        value_q / mlh.
      search_heads="blend":          (1 - q_weight) * value + q_weight * value_q WDLs / mlh."""

    MODES = ("search", "outcome", "q_value", "blend")

    def __init__(self, model, search_heads: str = "search", q_weight: float = 0.5):
        super().__init__()
        if search_heads not in self.MODES:
            raise ValueError(f"search_heads={search_heads!r}, expected one of {self.MODES}")
        self.model = model
        self.mode = search_heads
        self.q_weight = q_weight

    def forward(self, x):
        p, v, vq, mlh, mlh_q = self.model(x, return_all=True)
        if self.mode == "search":
            return p, torch.softmax(vq, dim=-1), mlh, mlh_q
        if self.mode == "q_value":
            return p, torch.softmax(vq, dim=-1), mlh, mlh
        if self.mode == "blend":
            w = self.q_weight
            return p, (1.0 - w) * torch.softmax(v, dim=-1) + w * torch.softmax(vq, dim=-1), mlh, mlh
        return p, torch.softmax(v, dim=-1), mlh, mlh
