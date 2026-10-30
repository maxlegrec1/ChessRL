"""Quantization experiment inputs (run on the box, project venv): calibration planes from the
benchmark positions (the C++ encoder, exactly what the engine sees) and random-weight checkpoints
of bigger encoders, for speed only."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import _inference_cpp as cpp  # noqa: E402

from src.models.factory import build_model  # noqa: E402

P = Path("/workspace/perf/quant")
P.mkdir(parents=True, exist_ok=True)

games = [
    line.split()
    for line in ROOT / "data/bench/positions.txt".read_text().splitlines()
    if line and not line.startswith("#")
]
planes = np.stack(
    [
        np.asarray(cpp.encode_position_with_history("", g), dtype=np.float32).reshape(8, 8, 111)
        for g in games
    ]
)
np.save(P / "calib.npy", planes)
print("calibration", planes.shape)

SIZES = {  # name: (d_model, layers, heads, d_ff)
    "enc192x6": (192, 6, 8, 384),
    "enc256x8": (256, 8, 8, 512),
    "enc384x10": (384, 10, 12, 768),
}
for name, (d, layers, heads, dff) in SIZES.items():
    spec = {"arch": "encoder", "d_model": d, "layers": layers, "heads": heads, "d_ff": dff}
    model = build_model(spec)
    params = sum(p.numel() for p in model.parameters())
    torch.save(
        {"model": model.state_dict(), "model_spec": spec, "config": {"search_heads": "outcome"}},
        P / f"{name}.pt",
    )
    print(name, f"{params / 1e6:.2f}M parameters")
