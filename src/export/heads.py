"""The checkpoint's `search_heads` setting: which heads the search reads."""

from __future__ import annotations

import torch


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
      search_heads="blend":          the average of the two WDLs / mlh."""

    MODES = ("search", "outcome", "q_value", "blend")

    def __init__(self, model, search_heads: str = "search"):
        super().__init__()
        if search_heads not in self.MODES:
            raise ValueError(f"search_heads={search_heads!r}, expected one of {self.MODES}")
        self.model = model
        self.mode = search_heads

    def forward(self, x):
        p, v, vq, mlh, mlh_q = self.model(x, return_all=True)
        if self.mode == "search":
            return p, torch.softmax(vq, dim=-1), mlh, mlh_q
        if self.mode == "q_value":
            return p, torch.softmax(vq, dim=-1), mlh, mlh
        if self.mode == "blend":
            return p, 0.5 * (torch.softmax(v, dim=-1) + torch.softmax(vq, dim=-1)), mlh, mlh
        return p, torch.softmax(v, dim=-1), mlh, mlh
