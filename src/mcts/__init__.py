from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CPP_DIR = _ROOT / "cpp"
if str(_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_DIR))

import _inference_cpp as _cpp  # type: ignore[attr-defined]


def _coerce_history(history: Iterable[str]) -> list[str]:
    return list(history)


@dataclass
class _BaseMCTSConfig:
    engine_path: str
    simulations: int
    cpuct: float
    temperature: float
    temperature_at_root: float


class _BaseMCTS:
    def __init__(
        self,
        engine_path: str,
        *,
        simulations: int = 800,
        cpuct: float = 1.5,
        temperature: float = 1.0,
        temperature_at_root: float = 0.0,
    ) -> None:
        self._config = _BaseMCTSConfig(
            engine_path=engine_path,
            simulations=simulations,
            cpuct=cpuct,
            temperature=temperature,
            temperature_at_root=temperature_at_root,
        )

    def _prepare_call(
        self,
        *,
        simulations: int | None,
        cpuct: float | None,
        temperature: float | None,
        temperature_at_root: float | None,
        engine_path: str | None,
    ) -> _BaseMCTSConfig:
        cfg = self._config
        return _BaseMCTSConfig(
            engine_path=engine_path or cfg.engine_path,
            simulations=simulations if simulations is not None else cfg.simulations,
            cpuct=cpuct if cpuct is not None else cfg.cpuct,
            temperature=temperature if temperature is not None else cfg.temperature,
            temperature_at_root=(
                temperature_at_root if temperature_at_root is not None else cfg.temperature_at_root
            ),
        )

    @staticmethod
    def _format_result(payload: Mapping[str, object]) -> dict[str, object]:
        action = payload.get("action")
        policy_pairs = payload.get("policy") or []
        q_values_raw = payload.get("q_values") or {}
        value = payload.get("value")

        policy: MutableMapping[str, float] = {}
        for entry in policy_pairs:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                move, prob = entry
                policy[str(move)] = float(prob)

        q_values = {str(move): float(q) for move, q in dict(q_values_raw).items()}

        return {
            "action": action,
            "policy": policy,
            "q_values": q_values,
            "value": tuple(value) if value is not None else None,
        }


class SmallMCTS(_BaseMCTS):
    def __init__(
        self,
        engine_path: str = "model.trt",
        *,
        simulations: int = 800,
        cpuct: float = 1.5,
        temperature: float = 1.0,
        temperature_at_root: float = 0.0,
    ) -> None:
        super().__init__(
            engine_path,
            simulations=simulations,
            cpuct=cpuct,
            temperature=temperature,
            temperature_at_root=temperature_at_root,
        )

    def run(
        self,
        fens: Sequence[str],
        *,
        simulations: int | None = None,
        cpuct: float | None = None,
        temperature: float | None = None,
        temperature_at_root: float | None = None,
        engine_path: str | None = None,
    ) -> list[dict[str, object]]:
        if isinstance(fens, str):
            raise TypeError("SmallMCTS.run requires a sequence of FEN strings, not a single string.")

        fen_list = [str(fen) for fen in fens]
        if not fen_list:
            return []

        cfg = self._prepare_call(
            simulations=simulations,
            cpuct=cpuct,
            temperature=temperature,
            temperature_at_root=temperature_at_root,
            engine_path=engine_path,
        )
        payloads = _cpp.small_mcts_search_batch(
            fen_list,
            cfg.simulations,
            cfg.cpuct,
            cfg.temperature,
            cfg.engine_path,
            cfg.temperature_at_root,
        )
        return [self._format_result(payload) for payload in payloads]

    __call__ = run


__all__ = [
    "SmallMCTS",
]

