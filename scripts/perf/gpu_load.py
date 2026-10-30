"""Synthetic trainer load for GPU-contention tests: trains the checkpoint's network on random
inputs (batch 1024, bf16 autocast, AdamW) for --seconds, busy for a --duty share of the time
(the real trainer, paced by self-play, is busy about 15% of the time)."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.factory import model_from_checkpoint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seconds", type=float, default=110)
ap.add_argument("--duty", type=float, default=0.15)
ap.add_argument("--batch", type=int, default=1024)
a = ap.parse_args()

ckpt = torch.load(a.ckpt, map_location="cuda", weights_only=False)
model = model_from_checkpoint(ckpt, "cuda").train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
x = torch.randn(a.batch, 8, 8, 111, device="cuda")
y = torch.randint(0, 1929, (a.batch,), device="cuda")
steps, busy, t0 = 0, 0.0, time.time()
while time.time() - t0 < a.seconds:
    s = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(model(x).float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    dt = time.time() - s
    busy += dt
    steps += 1
    if a.duty < 1.0:
        time.sleep(dt * (1.0 - a.duty) / a.duty)
wall = time.time() - t0
print(
    f"gpu_load: {steps} steps in {wall:.0f} s, {steps / wall:.2f} steps/s, busy {busy / wall:.0%}, "
    f"{busy / steps * 1000:.0f} ms/step"
)
