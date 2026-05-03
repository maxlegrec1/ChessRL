import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent

TENSORRT_ROOT = Path(os.environ.get("TENSORRT_ROOT", "/usr/local/TensorRT-10.14.1.48"))
CUDA_ROOT = Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda"))

ext_modules = [
    Pybind11Extension(
        "_inference_cpp",
        sources=[
            "bindings.cpp",
            "inference.cpp",
            "mcts/common.cpp",
            "mcts/mcts_small.cpp",
            "mcts/single_inference.cpp",
            "selfplay.cpp",
        ],
        include_dirs=[
            str(ROOT / "include"),
            str(ROOT / "chess-library/include"),
            str(TENSORRT_ROOT / "include"),
            str(CUDA_ROOT / "include"),
            str(ROOT),
        ],
        library_dirs=[
            str(TENSORRT_ROOT / "lib"),
            str(CUDA_ROOT / "lib64"),
        ],
        libraries=["nvinfer", "cudart"],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-DNDEBUG",
            "-std=c++17",
            "-DCHESSRL_MCTS_NO_STANDALONE",
        ],
    )
]

setup(
    name="chessrl_inference_cpp",
    version="0.1.0",
    description="Pybind11 bindings for the ChessRL TensorRT inference engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

