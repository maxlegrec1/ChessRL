from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class MetricsSnapshot:
    total_positions: int
    total_games: int
    positions_per_sec: float
    nodes_per_sec: float
    positions_per_hour: float
    buffer_fill_pct: float
    engine_path: str
    last_engine_swap: float | None


class MetricsTracker:
    """Thread-safe tracker for throughput and counters."""

    def __init__(self, *, simulations_per_position: int) -> None:
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._last_rate_sample = self._start
        self._rate_window = 2.0  # seconds
        self._ema_alpha = 0.2

        self._positions = 0
        self._games = 0
        self._positions_since_sample = 0
        self._positions_per_sec = 0.0
        self._simulations_per_position = simulations_per_position

        self.buffer_fill_pct = 0.0
        self.engine_path = "unknown"
        self.last_engine_swap: float | None = None

    # ------------------------------------------------------------------ #
    def record_positions(self, count: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._positions += count
            self._positions_since_sample += count
            elapsed = now - self._last_rate_sample
            if elapsed >= self._rate_window:
                inst = self._positions_since_sample / max(elapsed, 1e-6)
                if self._positions_per_sec == 0.0:
                    self._positions_per_sec = inst
                else:
                    self._positions_per_sec = (self._ema_alpha * inst) + (
                        (1 - self._ema_alpha) * self._positions_per_sec
                    )
                self._positions_since_sample = 0
                self._last_rate_sample = now

    def record_game_finished(self, count: int = 1) -> None:
        with self._lock:
            self._games += count

    def update_buffer_fill(self, fill_ratio: float) -> None:
        self.buffer_fill_pct = max(0.0, min(1.0, fill_ratio)) * 100.0

    def update_engine_path(self, path: str) -> None:
        self.engine_path = path
        self.last_engine_swap = time.time()

    # ------------------------------------------------------------------ #
    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            positions = self._positions
            games = self._games
            positions_per_sec = self._positions_per_sec if self._positions_per_sec > 0 else (
                positions / max(time.monotonic() - self._start, 1e-6)
            )

        nodes_per_sec = positions_per_sec * self._simulations_per_position
        positions_per_hour = positions_per_sec * 3600.0

        return MetricsSnapshot(
            total_positions=positions,
            total_games=games,
            positions_per_sec=positions_per_sec,
            nodes_per_sec=nodes_per_sec,
            positions_per_hour=positions_per_hour,
            buffer_fill_pct=self.buffer_fill_pct,
            engine_path=self.engine_path,
            last_engine_swap=self.last_engine_swap,
        )


__all__ = ["MetricsTracker", "MetricsSnapshot"]

