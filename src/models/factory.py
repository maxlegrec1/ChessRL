"""Build networks from a spec dict (stored in every checkpoint as `model_spec`)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

import torch

DEFAULT_SPEC = {"arch": "resnet", "blocks": 4, "channels": 64}


def normalize_spec(spec) -> dict:
    if spec is None:
        return dict(DEFAULT_SPEC)
    if is_dataclass(spec):
        spec = asdict(spec)
    spec = dict(spec)
    arch = spec.get("arch", "resnet")
    if arch == "resnet":
        return {
            "arch": "resnet",
            "blocks": int(spec.get("blocks", 4)),
            "channels": int(spec.get("channels", 64)),
        }
    if arch == "encoder":
        keys = ("d_model", "layers", "heads", "d_ff")
        return {"arch": "encoder", **{k: int(spec[k]) for k in keys}}
    raise ValueError(f"unknown arch {arch!r}")


def build_model(spec=None) -> torch.nn.Module:
    spec = normalize_spec(spec)
    if spec["arch"] == "resnet":
        from src.models.small.small_model import SmallModel

        return SmallModel(blocks=spec["blocks"], channels=spec["channels"])
    from src.models.encoder import EncoderModel

    return EncoderModel(**{k: v for k, v in spec.items() if k != "arch"})


def model_from_checkpoint(ckpt: dict, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Rebuild the network a checkpoint was saved from (no model_spec: the 4x64 ResNet)."""
    model = build_model(ckpt.get("model_spec")).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model
