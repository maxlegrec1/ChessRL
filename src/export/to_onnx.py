from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from src.models.small.fen_encoder import INPUT_PLANES
from src.models.small.small_model import SmallModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BT4 checkpoint to ONNX.")
    parser.add_argument(
        "--ckpt_path", required=True, type=Path, help="Path to training checkpoint."
    )
    parser.add_argument("--onnx_path", required=True, type=Path, help="Destination ONNX file.")
    parser.add_argument("--device", default="cuda", help="Device for export (default: cuda).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--batch", type=int, default=1, help="Dummy batch size.")
    return parser.parse_args()


def load_state_dict(ckpt_path: Path, device: str) -> dict:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


class ExportWrapper(torch.nn.Module):
    """Routes the `value_q` head (trained against MCTS-backed-up WDL) into the
    engine's "value" output and exposes both moves-left heads:
      - mlh:   raw plies/25 estimate, true target.
      - mlh_q: raw plies/25 estimate, MCTS-backed target. Used by C++ MCTS for
               both the depth-incremented backup (×25 ↦ raw plies) and the
               tiebreaker (tanh applied at the call site)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        p, _v, vq, mlh, mlh_q = self.model(x, return_all=True)
        return p, torch.softmax(vq, dim=-1), mlh, mlh_q


def export_model(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    base_model = SmallModel().to(device)
    state_dict = load_state_dict(args.ckpt_path, args.device)
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()

    model = ExportWrapper(base_model).to(device)
    model.eval()

    dummy_input = torch.randn(args.batch, 8, 8, INPUT_PLANES, device=device, dtype=torch.float32)
    onnx_path = args.onnx_path
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "policy": {0: "batch_size"},
        "value": {0: "batch_size"},
        "mlh": {0: "batch_size"},
        "mlh_q": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["policy", "value", "mlh", "mlh_q"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )

    verify_export(onnx_path)


def verify_export(path: Path) -> None:
    print(f"\nVerifying {path}...")
    model_onnx = onnx.load(path)
    input_shape = model_onnx.graph.input[0].type.tensor_type.shape.dim
    dim0 = "?"
    if len(input_shape) > 0:
        if input_shape[0].HasField("dim_param"):
            dim0 = input_shape[0].dim_param
        elif input_shape[0].HasField("dim_value"):
            dim0 = str(input_shape[0].dim_value)

    print(f"Input Dimension 0 is: '{dim0}'")
    if dim0 == "batch_size":
        print("✅ SUCCESS: Model is dynamic.")
    else:
        print(f"❌ WARNING: Model is static (Batch size fixed to {dim0}).")


if __name__ == "__main__":
    export_model(parse_args())
