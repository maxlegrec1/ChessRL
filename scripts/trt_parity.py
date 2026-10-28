"""TensorRT FP16 engine vs PyTorch FP32 on the same positions, per checkpoint.

Self-play, the ladder and the puzzles all run the FP16 engine; the trainer only
sees PyTorch. This checks that the two still agree.

    uv run python scripts/trt_parity.py ckpt.pt [ckpt.pt ...] [--n 256] [--fp32-norms]

Builds each engine exactly like scripts/build_trt.sh (FP16, batch profile 1/256/512)
into a temp dir, runs it on the first --n puzzle positions with trtexec, and
prints top-1 agreement, policy KL and value error against PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.export.to_onnx import ExportWrapper  # noqa: E402
from src.models.factory import build_model, normalize_spec  # noqa: E402
from src.models.small.fen_encoder import fen_to_tensor  # noqa: E402


def trt_run(onnx: Path, x: np.ndarray, work: Path, fp32_norms: bool) -> dict[str, np.ndarray]:
    engine = work / (onnx.stem + ("_fp32norm" if fp32_norms else "") + ".trt")
    n = x.shape[0]
    build = [
        "trtexec",
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        "--fp16",
        "--minShapes=input:1x8x8x111",
        "--optShapes=input:256x8x8x111",
        "--maxShapes=input:512x8x8x111",
        f"--shapes=input:{n}x8x8x111",
    ]
    if fp32_norms:
        # Keep every Pow / ReduceMean (the RMSNorm reductions) in FP32.
        build += [
            "--precisionConstraints=obey",
            "--layerPrecisions=*Pow*:fp32,*ReduceMean*:fp32,*Sqrt*:fp32",
        ]
    subprocess.run(build, check=True, capture_output=True)
    inp = work / "input.bin"
    x.astype(np.float32).tofile(inp)
    out = work / "out.json"
    subprocess.run(
        [
            "trtexec",
            f"--loadEngine={engine}",
            f"--shapes=input:{n}x8x8x111",
            f"--loadInputs=input:{inp}",
            f"--exportOutput={out}",
            "--iterations=1",
            "--warmUp=0",
            "--duration=0",
        ],
        check=True,
        capture_output=True,
    )
    res = {}
    for t in json.loads(out.read_text()):
        dims = [int(d) for d in str(t["dimensions"]).split("x")]
        res[t["name"]] = np.asarray(t["values"], dtype=np.float32).reshape(dims)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+", type=Path)
    p.add_argument("--n", type=int, default=256)
    p.add_argument(
        "--fp32-norms", action="store_true", help="also test an engine with norm reductions in FP32"
    )
    a = p.parse_args()
    if not 1 <= a.n <= 512:
        p.error("--n must be between 1 and 512 (the engine profile's max batch)")
    with open(ROOT / "data/puzzles/lichess_1k.csv") as fh:
        fens = [row["FEN"] for row in csv.DictReader(fh)][: a.n]
    x = np.stack([fen_to_tensor(f) for f in fens])
    for path in a.ckpts:
        ckpt = torch.load(path, map_location="cuda", weights_only=False)
        model = build_model(normalize_spec(ckpt["model_spec"])).cuda().eval()
        model.load_state_dict(ckpt["model"])
        heads = (ckpt.get("config") or {}).get("search_heads", "search")
        wrap = ExportWrapper(model, heads).eval()
        with torch.no_grad():
            ref = [t.float().cpu().numpy() for t in wrap(torch.from_numpy(x).cuda())]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            onnx = work / (path.stem + ".onnx")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.export.to_onnx",
                    "--ckpt_path",
                    str(path),
                    "--onnx_path",
                    str(onnx),
                ],
                check=True,
                capture_output=True,
                cwd=ROOT,
            )
            for fp32_norms in [False, True] if a.fp32_norms else [False]:
                out = trt_run(onnx, x, work, fp32_norms)
                pol, val = out["policy"], out["value"]
                bad = ~np.isfinite(pol).all(1) | ~np.isfinite(val).all(1)
                lp_ref = torch.log_softmax(torch.from_numpy(ref[0]), -1)
                lp_trt = torch.log_softmax(
                    torch.from_numpy(np.nan_to_num(pol, nan=0.0, posinf=0.0, neginf=0.0)), -1
                )
                kl = (lp_ref.exp() * (lp_ref - lp_trt)).sum(-1).numpy()
                top1 = (ref[0].argmax(1) == np.nan_to_num(pol).argmax(1)).mean()
                verr = np.abs(np.nan_to_num(val) - ref[1]).sum(1)
                tag = "fp16+fp32norms" if fp32_norms else "fp16 (as deployed)"
                print(
                    f"{path.name:22s} {tag:20s} non-finite {bad.mean():6.1%}  top-1 agree {top1:6.1%}  "
                    f"policy KL mean {kl.mean():.4f} p95 {np.quantile(kl, 0.95):.4f}  "
                    f"value |dWDL| mean {verr.mean():.4f} max {verr.max():.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
