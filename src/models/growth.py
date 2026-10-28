"""Function-preserving growth of the ResNet trunk and of the encoder's depth.

Depth: new residual blocks get a zero-initialised final BatchNorm (gamma =
beta = 0), so each new block adds 0 to its input; block inputs are post-ReLU
(non-negative), so relu(x + 0) = x and the block is the identity.

Width: new channels compute live random features, but every weight that would
READ a new channel into an old channel or a head starts at 0 (old block
internals <- new stream, old block outputs <- new internals, heads <- new
stream). The old computation is untouched; those zero weights get gradient on
the first step (like a zero-initialised LoRA adapter). Pinning the new
channels themselves to 0 instead would leave them dead: ReLU(0) passes no
gradient.

Outputs are unchanged at the moment of growth in both train and eval mode.
"""

from __future__ import annotations

import torch

from src.models.small.small_model import SmallModel

_HEAD_STREAM_CONVS = ("policy_conv", "value_conv", "value_conv_q", "mlh_conv", "mlh_q_conv")


def _copy_bn(dst: torch.nn.BatchNorm2d, src: torch.nn.BatchNorm2d, *, zero_new: bool) -> None:
    c = src.num_features
    with torch.no_grad():
        if zero_new:
            dst.weight.zero_()
            dst.bias.zero_()
        dst.weight[:c] = src.weight
        dst.bias[:c] = src.bias
        dst.running_mean[:c] = src.running_mean
        dst.running_var[:c] = src.running_var
        dst.num_batches_tracked.copy_(src.num_batches_tracked)


@torch.no_grad()
def grow_resnet(old: SmallModel, blocks: int, channels: int) -> SmallModel:
    """Return a SmallModel(blocks, channels) computing the same function as `old`."""
    c0, b0 = old.channels, old.num_blocks
    if channels < c0 or blocks < b0:
        raise ValueError(f"can only grow: {b0}x{c0} -> {blocks}x{channels}")
    device = next(old.parameters()).device
    new = SmallModel(blocks=blocks, channels=channels).to(device)

    # Stem: old channels copied, new channels are fresh random features.
    new.input_conv.weight[:c0] = old.input_conv.weight
    _copy_bn(new.input_bn, old.input_bn, zero_new=False)

    # Existing blocks.
    for nb, ob in zip(new.blocks[:b0], old.blocks, strict=True):
        nb.conv1.weight[:c0, :c0] = ob.conv1.weight
        nb.conv1.weight[:c0, c0:] = 0.0  # old internals <- new stream channels
        _copy_bn(nb.bn1, ob.bn1, zero_new=False)
        nb.conv2.weight[:c0, :c0] = ob.conv2.weight
        nb.conv2.weight[:c0, c0:] = 0.0  # old outputs <- new internal channels
        _copy_bn(nb.bn2, ob.bn2, zero_new=False)

    # New blocks are identities until they learn.
    for nb in new.blocks[b0:]:
        nb.bn2.weight.zero_()
        nb.bn2.bias.zero_()

    # Heads: copy everything; stream-reading 1x1 convs ignore new channels.
    old_state = old.state_dict()
    for name, tensor in new.state_dict().items():
        if name.startswith(("input_", "blocks.")):
            continue
        src = old_state[name]
        head = name.split(".")[0]
        if head in _HEAD_STREAM_CONVS and name.endswith("weight"):
            tensor.zero_()
            tensor[:, :c0] = src
        else:
            tensor.copy_(src)
    new.train(old.training)
    return new


@torch.no_grad()
def grow_encoder(old, layers: int):
    """Return an EncoderModel with `layers` blocks computing the same function as `old`.

    New blocks are appended before the final norm with their attention output
    and feed-forward down projections (and x0 gain) at zero, so each starts as
    the identity; those weights get gradient on the first step."""
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
