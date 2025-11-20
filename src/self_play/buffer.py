from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class SelfPlayRecord:
    """Single training datum produced by self-play."""

    fen: str
    value: float
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
        self._values: List[float] = []
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
            self._values = [float(v) for v in table.column("value").to_pylist()]
            self._policies = table.column("policy").to_pylist()
            self._pending_flushes = 0
            self._random_ready = len(self._fens) >= self.capacity
            return len(self._fens)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_records(self, records: Sequence[SelfPlayRecord]) -> None:
        """Insert a batch of records, sequentially until full then random replace."""
        if not records:
            return

        with self._lock:
            for record in records:
                self._insert_locked(record)

            self._pending_flushes += 1
            if self._pending_flushes >= self._flush_interval:
                self._flush_locked()

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
    def _insert_locked(self, record: SelfPlayRecord) -> None:
        if self.size < self.capacity:
            self._fens.append(record.fen)
            self._values.append(float(record.value))
            self._policies.append(record.policy)
            if self.size == self.capacity:
                self._shuffle_locked()
                self._random_ready = True
            return

        if not self._random_ready:
            self._shuffle_locked()
            self._random_ready = True

        idx = random.randrange(self.capacity)
        self._fens[idx] = record.fen
        self._values[idx] = float(record.value)
        self._policies[idx] = record.policy

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
                "value": pa.array(self._values, type=pa.float32()),
                "policy": pa.array(self._policies, type=pa.string()),
            }
        )
        pq.write_table(table, self.path)
        self._pending_flushes = 0


__all__ = ["SelfPlayBuffer", "SelfPlayRecord"]

