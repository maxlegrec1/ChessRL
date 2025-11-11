from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent

TENSORRT_ROOT = Path("/usr/local/TensorRT-10.14.1.48")
CUDA_ROOT = Path("/usr/local/cuda")

ext_modules = [
    Pybind11Extension(
        "_inference_cpp",
        sources=["bindings.cpp", "inference.cpp"],
        include_dirs=[
            str(ROOT / "include"),
            str(ROOT / "chess-library/include"),
            str(TENSORRT_ROOT / "include"),
            str(CUDA_ROOT / "include"),
        ],
        library_dirs=[
            str(TENSORRT_ROOT / "lib"),
            str(CUDA_ROOT / "lib64"),
        ],
        libraries=["nvinfer", "cudart"],
        extra_compile_args=["-O3", "-std=c++17"],
    )
]

setup(
    name="chessrl_inference_cpp",
    version="0.1.0",
    description="Pybind11 bindings for the ChessRL TensorRT inference engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

