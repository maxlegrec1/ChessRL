import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent

TENSORRT_ROOT_DEFAULT = "/usr/local/TensorRT-10.14.1.48"
TENSORRT_ROOT = Path(os.environ.get("TENSORRT_ROOT", TENSORRT_ROOT_DEFAULT))
CUDA_ROOT = Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda"))

# Fall back to a system (apt) TensorRT install if the tarball layout is absent.
# apt ships headers under /usr/include (multi-arch) and libs under /lib/x86_64-linux-gnu.
_trt_include_dirs: list[str] = []
_trt_library_dirs: list[str] = []
if (TENSORRT_ROOT / "include" / "NvInfer.h").exists():
    _trt_include_dirs.append(str(TENSORRT_ROOT / "include"))
    _trt_library_dirs.append(str(TENSORRT_ROOT / "lib"))
# Always add common system paths so multi-arch / apt installs work too. The
# compiler will skip directories that don't exist.
_trt_include_dirs.append("/usr/include/x86_64-linux-gnu")
_trt_include_dirs.append("/usr/include")
_trt_library_dirs.append("/usr/lib/x86_64-linux-gnu")

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
            *_trt_include_dirs,
            str(CUDA_ROOT / "include"),
            str(ROOT),
        ],
        library_dirs=[
            *_trt_library_dirs,
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

