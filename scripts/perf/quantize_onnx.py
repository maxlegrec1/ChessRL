"""Post-training quantization of an ONNX net with NVIDIA ModelOpt (run in its own venv):
python quantize_onnx.py IN.onnx int8|fp8 calib.npy OUT.onnx"""

import sys

import numpy as np
from modelopt.onnx.quantization import quantize

src, mode, calib, out = sys.argv[1:5]
data = {"input": np.load(calib)}
kwargs = dict(
    onnx_path=src,
    quantize_mode=mode,
    calibration_data=data,
    calibration_method="entropy" if mode == "int8" else "max",
    output_path=out,
)
try:
    quantize(**kwargs, calibration_eps=["cpu"])
except TypeError:  # older ModelOpt without calibration_eps
    quantize(**kwargs)
print("quantized", src, "->", out, mode)
