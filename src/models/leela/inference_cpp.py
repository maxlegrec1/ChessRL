from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import importlib
import sys

_BINDING = None

ValueVector = Tuple[float, float, float]
LeelaPolicyResult = Tuple[List[str], List[ValueVector], List[ValueVector]]


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


class Leelacpp:
    """
    Python wrapper around the Leela TensorRT C++ inference.
    """

    def __init__(
        self,
        default_temperature: float = 0.0,
        default_iterations: int = 1,
        engine_path: str = "leela.trt",
    ) -> None:
        if default_iterations < 1:
            raise ValueError("default_iterations must be >= 1")
        self._binding = _load_binding()
        self._default_temperature = float(default_temperature)
        self._default_iterations = int(default_iterations)
        self._engine_path = str(engine_path)

    def __call__(
        self,
        origin_fens: Sequence[str],
        move_lists: Sequence[Sequence[str]],
        temperature: float | None = None,
        iterations: int | None = 1,
        return_timings: bool = False,
    ) -> LeelaPolicyResult | Tuple[LeelaPolicyResult, InferenceStats]:
        if not origin_fens or not move_lists:
            empty: LeelaPolicyResult = ([], [], [])
            if return_timings:
                return empty, InferenceStats(0.0, 0.0, 0.0, 0.0)
            return empty

        if len(origin_fens) != len(move_lists):
            raise ValueError("origin_fens and move_lists must have the same length.")

        temp = self._default_temperature if temperature is None else float(temperature)
        iters = self._default_iterations if iterations is None else int(iterations)
        if iters < 1:
            raise ValueError("iterations must be >= 1")

        fen_list = list(origin_fens)
        move_lists_converted = [list(seq) for seq in move_lists]
        result = self._binding.leela_generate_from_move_lists(
            fen_list, move_lists_converted, temp, iters, self._engine_path
        )

        moves = list(result.moves)
        value_winner = [tuple(map(float, vec)) for vec in result.value_winner]
        value_q = [tuple(map(float, vec)) for vec in result.value_q]
        policy_result: LeelaPolicyResult = (moves, value_winner, value_q)

        if return_timings:
            timings = result.timings
            stats = InferenceStats(
                encode_ms=timings.encode_ms,
                inference_ms=timings.inference_ms,
                sampling_ms=timings.sampling_ms,
                total_ms=timings.total_ms,
            )
            return policy_result, stats
        return policy_result


__all__ = ["Leelacpp", "InferenceStats"]

