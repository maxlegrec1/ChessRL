from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional


class EngineWatcher:
    """Polls the `.iteration` file for the latest engine path."""

    def __init__(self, iteration_file: Path, *, poll_interval: float = 600.0) -> None:
        self.iteration_file = iteration_file
        self.poll_interval = poll_interval
        self._last_check = 0.0
        self._last_mtime = 0.0
        self._engine_path: str = "artifacts/engines/model_dynamic_initial.trt"

        # Load initial state if possible.
        self._refresh(force=True)

    @property
    def engine_path(self) -> str:
        return self._engine_path

    def maybe_refresh(self) -> Optional[str]:
        """Refresh if poll interval elapsed; returns new engine path if changed."""
        now = time.monotonic()
        if (now - self._last_check) < self.poll_interval:
            return None
        self._last_check = now
        changed = self._refresh()
        if changed:
            return self._engine_path
        return None

    # ------------------------------------------------------------------ #
    def _refresh(self, *, force: bool = False) -> bool:
        try:
            stat = self.iteration_file.stat()
        except FileNotFoundError:
            return False

        if not force and stat.st_mtime <= self._last_mtime:
            return False

        self._last_mtime = stat.st_mtime
        self._last_check = time.monotonic()
        new_engine = self._read_latest_engine()
        if new_engine and new_engine != self._engine_path:
            self._engine_path = new_engine
            return True
        if force and new_engine:
            self._engine_path = new_engine
        return False

    def _read_latest_engine(self) -> Optional[str]:
        try:
            with self.iteration_file.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, skipinitialspace=True)
                last_row = None
                for row in reader:
                    last_row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        except FileNotFoundError:
            return None

        if not last_row:
            return None
        return (last_row.get("engine_path") or "").strip() or None


__all__ = ["EngineWatcher"]

