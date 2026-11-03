"""Function-preserving growth of the encoder's depth (growth.schedule in the config).

New blocks are appended before the final norm with their attention output and feed-forward down
projections (and x0 gain) at zero, so each starts as the identity and the network's outputs are
unchanged at the moment of growth; those weights get gradient on the first step.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def grow_encoder(old, layers: int):
    """Return an EncoderModel with `layers` blocks computing the same function as `old`."""
    from src.models.encoder import EncoderModel

    cfg = dict(old.cfg)
    if layers < cfg["layers"]:
        raise ValueError(f"can only grow: {cfg['layers']} -> {layers} layers")
    device = next(old.parameters()).device
    new = EncoderModel(**{**cfg, "layers": layers}).to(device)
    new.load_state_dict(old.state_dict(), strict=False)  # shared names: embed, blocks.0..n-1, heads
    for block in new.blocks[cfg["layers"] :]:
        block.out.weight.zero_()
        block.down.weight.zero_()
        block.x0_gain.zero_()
    new.train(old.training)
    return new
