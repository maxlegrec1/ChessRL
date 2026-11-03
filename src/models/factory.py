"""Build networks from a spec dict (stored in every checkpoint as `model_spec`)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

import torch

DEFAULT_SPEC = {"arch": "encoder", "d_model": 128, "layers": 4, "heads": 8, "d_ff": 256}


def normalize_spec(spec) -> dict:
    if spec is None:
        return dict(DEFAULT_SPEC)
    if is_dataclass(spec):
        spec = asdict(spec)
    spec = dict(spec)
    arch = spec.get("arch")
    if arch != "encoder":
        # The round-1 ResNet (checkpoints without a spec, or arch "resnet") is gone: the fused
        # self-play network and the canonical 1,858-move policy are encoder-only.
        raise ValueError(f"unsupported arch {arch!r}: only the encoder is supported")
    keys = ("d_model", "layers", "heads", "d_ff")
    return {"arch": "encoder", **{k: int(spec.get(k, DEFAULT_SPEC[k])) for k in keys}}


def build_model(spec=None) -> torch.nn.Module:
    from src.models.encoder import EncoderModel

    spec = normalize_spec(spec)
    return EncoderModel(**{k: v for k, v in spec.items() if k != "arch"})


def model_from_checkpoint(ckpt: dict, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Rebuild the network a checkpoint was saved from."""
    if "model_spec" not in ckpt:
        raise ValueError("checkpoint without model_spec: a round-1 ResNet, no longer supported")
    model = build_model(ckpt["model_spec"]).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    return model
