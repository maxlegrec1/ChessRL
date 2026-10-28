"""Self-play and trainer throughput as a function of network size.

For each spec: random-init checkpoint -> ONNX -> TensorRT engine, then
scripts/bench_selfplay.py with production settings, then a trainer
forward+backward timing at the production batch size. Run on an idle GPU.

  uv run python scripts/bench_sizes.py --specs 4x64 6x96 8x128 10x128 12x160 \\
      --out artifacts/perf/sizes.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.factory import build_model  # noqa: E402


def parse_spec(text: str) -> dict:
    if text.startswith("enc"):
        # enc<d_model>x<layers>, e.g. enc128x4 (heads = d_model / 16, d_ff = 2 * d_model)
        d, layers = (int(v) for v in text[3:].split("x"))
        return {
            "arch": "encoder",
            "d_model": d,
            "layers": layers,
            "heads": max(1, d // 16),
            "d_ff": 2 * d,
        }
    blocks, channels = (int(v) for v in text.split("x"))
    return {"arch": "resnet", "blocks": blocks, "channels": channels}


def trainer_samples_per_sec(spec: dict, batch: int = 2048, iters: int = 20) -> float:
    model = build_model(spec).cuda().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.rand(batch, 8, 8, 111, device="cuda")
    for i in range(iters + 5):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x, return_all=True)
            loss = sum(o.float().pow(2).mean() for o in out)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    return batch * iters / (time.time() - t0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--specs", nargs="+", default=["4x64", "6x96", "8x128", "10x128", "12x160"])
    p.add_argument("--duration-sec", type=float, default=60)
    p.add_argument("--warmup-sec", type=float, default=20)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "perf" / "sizes.json")
    a = p.parse_args()
    scratch = ROOT / "artifacts" / "perf" / "scratch_sizes"
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for text in a.specs:
        spec = parse_spec(text)
        model = build_model(spec)
        params = sum(t.numel() for t in model.parameters())
        ckpt = scratch / f"{text}.pt"
        torch.save({"model": model.state_dict(), "model_spec": spec}, ckpt)
        subprocess.run(
            [
                "bash",
                str(ROOT / "scripts" / "build_trt.sh"),
                str(ckpt),
                "0",
                str(scratch),
                str(scratch / ".iteration"),
            ],
            check=True,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
        )
        engine = scratch / f"{text}.trt"
        out_json = scratch / f"bench_{text}.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bench_selfplay.py"),
                "--engine",
                str(engine),
                "--duration-sec",
                str(a.duration_sec),
                "--warmup-sec",
                str(a.warmup_sec),
                "--threads",
                str(a.threads),
                "--virtual-loss-width",
                "8",
                "--random-opening-plies",
                "8",
                "--dirichlet-alpha",
                "0.3",
                "--dirichlet-epsilon",
                "0.25",
                "--out",
                str(out_json),
            ],
            check=True,
            cwd=ROOT,
        )
        bench = json.loads(out_json.read_text())
        row = {
            "spec": text,
            "params": params,
            "trainer_samples_per_sec": trainer_samples_per_sec(spec),
            "selfplay": bench,
        }
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "selfplay"}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
