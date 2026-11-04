import os
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext as _pybind_build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent

CUDA_ROOT = Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda"))

# CUDA compute capability for the fused network kernel (mma.sync m16n8k16: sm_80 or
# newer). Default sm_90 (H200/Hopper). Override via env: sm_89 for 4090/Ada, sm_86 for
# 3090/Ampere, sm_80 for A100.
# Mismatch with the target GPU is silently accepted by nvcc but the kernel
# will fail at runtime with cryptic CUDA errors — so we make this explicit.
CUDA_ARCH = os.environ.get("CHESSRL_CUDA_ARCH", "sm_90")
if not CUDA_ARCH.startswith("sm_") or not CUDA_ARCH[3:].isdigit():
    raise RuntimeError(
        f"CHESSRL_CUDA_ARCH={CUDA_ARCH!r} is malformed; expected sm_<digits>, "
        "e.g. sm_90 (H200/Hopper), sm_89 (4090/Ada), sm_86 (3090/Ampere), "
        "sm_80 (A100)."
    )
_CUDA_COMPUTE = "compute_" + CUDA_ARCH[3:]

ext_modules = [
    Pybind11Extension(
        "_inference_cpp",
        sources=[
            "bindings.cpp",
            "policy_vocab.cpp",
            "mcts/common.cpp",
            "mcts/mcts_small.cpp",
            "mcts/single_inference.cpp",
            "mcts/small_encoding.cpp",
            "net/encoder_net.cpp",
            "selfplay.cpp",
        ],
        include_dirs=[
            str(ROOT / "include"),
            str(ROOT / "chess-library/include"),
            str(CUDA_ROOT / "include"),
            str(ROOT),
        ],
        library_dirs=[
            str(CUDA_ROOT / "lib64"),
        ],
        libraries=["cudart"],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-DNDEBUG",
            "-std=c++17",
            "-DCHESSRL_MCTS_NO_STANDALONE",
        ],
    )
]

CU_SOURCES = ["net/encoder_kernel.cu"]


class build_ext(_pybind_build_ext):
    """pybind11 build_ext + an nvcc pre-pass for the fused network kernel.

    The project is otherwise pure g++; nvcc only ever sees net/encoder_kernel.cu and
    the plain headers it includes (deliberately free of chess-library). The compiled
    object is linked into the extension via extra_objects.
    """

    def build_extensions(self):
        nvcc = os.environ.get("NVCC") or str(CUDA_ROOT / "bin" / "nvcc")
        obj_dir = Path(self.build_temp) / "cu"
        obj_dir.mkdir(parents=True, exist_ok=True)
        for ext in self.extensions:
            for cu in CU_SOURCES:
                src = ROOT / cu
                obj = obj_dir / (Path(cu).stem + ".o")
                cmd = [
                    nvcc, "-c", str(src), "-o", str(obj),
                    "-O3", "-std=c++17",
                    "-Xcompiler", "-fPIC",
                    # Compute arch comes from CHESSRL_CUDA_ARCH (default sm_90 / H200).
                    # Override to match your GPU; see CHESSRL_CUDA_ARCH at the top of
                    # this file for the supported values.
                    "-gencode", f"arch={_CUDA_COMPUTE},code={CUDA_ARCH}",
                    f"-I{ROOT}",
                    f"-I{ROOT / 'include'}",
                    f"-I{CUDA_ROOT / 'include'}",
                ]
                print("[nvcc]", " ".join(cmd))
                subprocess.run(cmd, check=True)
                ext.extra_objects = list(ext.extra_objects or []) + [str(obj)]
        super().build_extensions()


setup(
    name="chessrl_inference_cpp",
    version="0.1.0",
    description="Pybind11 bindings for the ChessRL search, self-play and fused CUDA network",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

