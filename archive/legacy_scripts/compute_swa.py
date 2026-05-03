#!/usr/bin/env python3
"""Compute a weighted SWA over checkpoints/step_*.pt and write SWA.pt."""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict

import torch


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_PATH = ROOT / "SWA.pt"

STEP_RE = re.compile(r"step_(\d+)\.pt$")
# Small floor so earliest checkpoints contribute almost zero.
MIN_WEIGHT_RATIO = 1e-4


def _extract_weight(path: Path) -> int | None:
    match = STEP_RE.search(path.name)
    return int(match.group(1)) if match else None


def main() -> None:
    checkpoint_paths = sorted(
        p for p in CHECKPOINT_DIR.glob("step_*.pt") if _extract_weight(p) is not None
    )

    if not checkpoint_paths:
        raise SystemExit(f"No checkpoints matching step_*.pt found in {CHECKPOINT_DIR}")

    steps = [_extract_weight(p) for p in checkpoint_paths]
    assert all(s is not None for s in steps)
    step_vals = [int(s) for s in steps if s is not None]

    max_step = max(step_vals)
    min_step = min(step_vals)
    if max_step == min_step:
        weight_func = lambda step: 1.0  # noqa: E731
    else:
        decay = math.log(MIN_WEIGHT_RATIO) / (min_step - max_step)
        weight_func = lambda step: math.exp(decay * (step - max_step))  # noqa: E731

    weighted_sum: Dict[str, torch.Tensor] = {}
    original_dtypes: Dict[str, torch.dtype] = {}
    total_weight = 0.0

    for path, step in zip(checkpoint_paths, step_vals):
        weight = weight_func(step)

        print(f"Loading {path} (step={step}, weight={weight:.6e})")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "model" not in checkpoint:
            raise KeyError(f"'model' key missing in checkpoint {path}")

        state_dict = checkpoint["model"]
        if not weighted_sum:
            # First checkpoint initializes the accumulator.
            for name, tensor in state_dict.items():
                weighted_sum[name] = tensor.detach().to(torch.float32) * weight
                original_dtypes[name] = tensor.dtype
            total_weight += weight
            continue

        for name, tensor in state_dict.items():
            if name not in weighted_sum:
                raise KeyError(
                    f"Parameter '{name}' not present in earlier checkpoints; cannot average."
                )
            weighted_sum[name] += tensor.detach().to(torch.float32) * weight

        total_weight += weight

    averaged_state = {
        name: (tensor / total_weight).to(original_dtypes[name])
        for name, tensor in weighted_sum.items()
    }

    torch.save({"model": averaged_state}, OUTPUT_PATH)
    print(
        f"Saved SWA model to {OUTPUT_PATH} using {len(checkpoint_paths)} checkpoints "
        f"(total weight {total_weight})."
    )


if __name__ == "__main__":
    main()

