from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import importlib
import sys


_BINDING = None


def _load_binding():
    global _BINDING
    if _BINDING is None:
        cpp_dir = Path(__file__).resolve().parents[3] / "cpp"
        if not cpp_dir.exists():
            raise RuntimeError(f"Expected cpp directory at {cpp_dir}")
        if str(cpp_dir) not in sys.path:
            sys.path.insert(0, str(cpp_dir))
        _BINDING = importlib.import_module("_inference_cpp")
    return _BINDING


@dataclass(frozen=True)
class InferenceStats:
    encode_ms: float
    inference_ms: float
    sampling_ms: float
    total_ms: float


class CppPolicy:
    """
    Thin Python wrapper around the optimized C++ TensorRT policy inference.
    """

    def __init__(
        self,
        default_temperature: float = 0.0,
        default_iterations: int = 1,
        engine_path: str = "model.trt",
    ) -> None:
        if default_iterations < 1:
            raise ValueError("default_iterations must be >= 1")
        self._binding = _load_binding()
        self._default_temperature = float(default_temperature)
        self._default_iterations = int(default_iterations)
        self._engine_path = str(engine_path)

    def __call__(
        self,
        fens: Sequence[str],
        temperature: float | None = None,
        iterations: int | None = 1,
        return_timings: bool = False,
    ) -> List[str] | Tuple[List[str], InferenceStats]:
        if not fens:
            return [] if not return_timings else ([], InferenceStats(0.0, 0.0, 0.0, 0.0))

        temp = self._default_temperature if temperature is None else float(temperature)
        iters = self._default_iterations if iterations is None else int(iterations)
        if iters < 1:
            raise ValueError("iterations must be >= 1")

        result = self._binding.generate_moves(list(fens), temp, iters, self._engine_path)
        if return_timings:
            timings = result.timings
            stats = InferenceStats(
                encode_ms=timings.encode_ms,
                inference_ms=timings.inference_ms,
                sampling_ms=timings.sampling_ms,
                total_ms=timings.total_ms,
            )
            return list(result.moves), stats
        return list(result.moves)


__all__ = ["CppPolicy", "InferenceStats"]

