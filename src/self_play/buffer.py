from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class SelfPlayRecord:
    """Single training datum produced by self-play."""

    fen: str
    value: Tuple[float, float, float]
    policy: str


class SelfPlayBuffer:
    """Parquet-backed replay buffer with sequential fill then random replacement."""

    def __init__(
        self,
        path: Path,
        *,
        capacity: int = 100_000,
        flush_interval: int = 1,
    ) -> None:
        self.path = path
        self.capacity = capacity
        self._flush_interval = max(1, flush_interval)

        self._fens: List[str] = []
        self._values: List[Tuple[float, float, float]] = []
        self._policies: List[str] = []
        self._lock = threading.Lock()

        self._pending_flushes = 0
        self._random_ready = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> int:
        """Load existing buffer from disk if present; returns row count."""
        with self._lock:
            if not self.path.exists():
                self._fens = []
                self._values = []
                self._policies = []
                self._random_ready = False
                self._pending_flushes = 0
                return 0

            table = pq.read_table(self.path)
            self._fens = table.column("fen").to_pylist()
            if "value" in table.column_names:
                raw_values = table.column("value").to_pylist()
                self._values = [
                    _ensure_wdl_tuple(v) for v in raw_values  # type: ignore[arg-type]
                ]
            else:
                # Backwards compatibility: look for split columns
                vb = table.column("value_black").to_pylist()
                vd = table.column("value_draw").to_pylist()
                vw = table.column("value_white").to_pylist()
                self._values = [
                    (
                        float(vb[i]),
                        float(vd[i]),
                        float(vw[i]),
                    )
                    for i in range(len(vb))
                ]
            self._policies = table.column("policy").to_pylist()
            self._pending_flushes = 0
            self._random_ready = len(self._fens) >= self.capacity
            return len(self._fens)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_records(self, records: Sequence[SelfPlayRecord]) -> Sequence[int]:
        """Insert a batch of records, sequentially until full then random replace.

        Returns a list of row indices that were written, for downstream consumers.
        """
        if not records:
            return []

        indices: list[int] = []
        with self._lock:
            for record in records:
                idx = self._insert_locked(record)
                indices.append(idx)

            self._pending_flushes += 1
            if self._pending_flushes >= self._flush_interval:
                self._flush_locked()
        return indices

    def add_record(self, record: SelfPlayRecord) -> None:
        self.add_records((record,))

    @property
    def size(self) -> int:
        return len(self._fens)

    @property
    def full(self) -> bool:
        return self.size >= self.capacity

    def fill_ratio(self) -> float:
        if self.capacity == 0:
            return 0.0
        return min(1.0, self.size / self.capacity)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _insert_locked(self, record: SelfPlayRecord) -> int:
        if self.size < self.capacity:
            self._fens.append(record.fen)
            self._values.append(tuple(map(float, record.value)))
            self._policies.append(record.policy)
            if self.size == self.capacity:
                self._shuffle_locked()
                self._random_ready = True
            return self.size - 1

        if not self._random_ready:
            self._shuffle_locked()
            self._random_ready = True

        idx = random.randrange(self.capacity)
        self._fens[idx] = record.fen
        self._values[idx] = tuple(map(float, record.value))
        self._policies[idx] = record.policy
        return idx

    def _shuffle_locked(self) -> None:
        if self.size <= 1:
            return
        indices = list(range(self.size))
        random.shuffle(indices)
        self._fens = [self._fens[i] for i in indices]
        self._values = [self._values[i] for i in indices]
        self._policies = [self._policies[i] for i in indices]
        self._pending_flushes = 0
        self._flush_locked()

    def _flush_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "fen": pa.array(self._fens, type=pa.string()),
                "value": pa.array(
                    self._values,
                    type=pa.list_(pa.float32(), 3),
                ),
                "policy": pa.array(self._policies, type=pa.string()),
            }
        )
        pq.write_table(table, self.path)
        self._pending_flushes = 0


def _ensure_wdl_tuple(value_obj) -> Tuple[float, float, float]:
    if value_obj is None:
        return (0.0, 0.0, 0.0)
    if isinstance(value_obj, (list, tuple)):
        vals = list(value_obj)
        if len(vals) == 3:
            return (float(vals[0]), float(vals[1]), float(vals[2]))
        if len(vals) == 2:
            return (float(vals[0]), float(vals[1]), 0.0)
        if len(vals) == 1:
            v = float(vals[0])
            return (v, 0.0, 0.0)
    try:
        scalar = float(value_obj)
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)
    return (scalar, 0.0, 0.0)


__all__ = ["SelfPlayBuffer", "SelfPlayRecord"]

