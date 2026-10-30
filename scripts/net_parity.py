"""Fused CUDA network (.cnet) vs PyTorch FP32 on the same positions, per checkpoint.

Self-play, the ladder and the puzzles run the fused kernel; the trainer only sees
PyTorch. This checks that the two agree, with the metrics of the TensorRT check this
replaced: top-1 agreement, policy KL and value error, on the first --n puzzle positions.
It also runs the same positions through the packed input path (as self-play packs them)
and checks that it gives the dense path's outputs exactly.

    uv run python scripts/net_parity.py ckpt.pt [ckpt.pt ...] [--n 256]
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _inference_cpp  # noqa: E402

from src.export.to_cnet import export  # noqa: E402
from src.export.to_onnx import ExportWrapper  # noqa: E402
from src.models.factory import model_from_checkpoint  # noqa: E402
from src.models.small.fen_encoder import fen_to_tensor  # noqa: E402


def compare(ref: list[np.ndarray], out: dict[str, np.ndarray]) -> str:
    pol, val = out["policy"], out["value"]
    bad = ~np.isfinite(pol).all(1) | ~np.isfinite(val).all(1)
    lp_ref = torch.log_softmax(torch.from_numpy(ref[0]), -1)
    lp_net = torch.log_softmax(torch.from_numpy(np.nan_to_num(pol, nan=0.0, posinf=0.0, neginf=0.0)), -1)
    kl = (lp_ref.exp() * (lp_ref - lp_net)).sum(-1).numpy()
    top1 = (ref[0].argmax(1) == np.nan_to_num(pol).argmax(1)).mean()
    verr = np.abs(np.nan_to_num(val) - ref[1]).sum(1)
    merr = np.abs(out["mlh"] - ref[2]).max()
    return (
        f"non-finite {bad.mean():6.1%}  top-1 agree {top1:6.1%}  "
        f"policy KL mean {kl.mean():.5f} p95 {np.quantile(kl, 0.95):.5f}  "
        f"value |dWDL| mean {verr.mean():.4f} max {verr.max():.3f}  mlh max err {merr:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+", type=Path)
    p.add_argument("--n", type=int, default=256)
    a = p.parse_args()
    with open(ROOT / "data/puzzles/lichess_1k.csv") as fh:
        fens = [row["FEN"] for row in csv.DictReader(fh)][: a.n]
    x = np.stack([fen_to_tensor(f) for f in fens]).astype(np.float32)
    for path in a.ckpts:
        ckpt = torch.load(path, map_location="cuda", weights_only=False)
        model = model_from_checkpoint(ckpt, "cuda").eval()
        heads = (ckpt.get("config") or {}).get("search_heads", "search")
        with torch.no_grad():
            ref = [t.float().cpu().numpy() for t in ExportWrapper(model, heads)(torch.from_numpy(x).cuda())]
        with tempfile.TemporaryDirectory() as tmp:
            net = Path(tmp) / (path.stem + ".cnet")
            export(path, net)
            dense = _inference_cpp.fused_forward_dense(str(net), x)
            packed = _inference_cpp.fused_forward_positions(str(net), fens, [[] for _ in fens])
        same = all(np.array_equal(dense[k], packed[k]) for k in dense)
        print(f"{path.name:22s} fused FP16  {compare(ref, dense)}  packed == dense: {same}", flush=True)


if __name__ == "__main__":
    main()
