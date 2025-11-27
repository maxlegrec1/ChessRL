from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq

from src.models.small.fen_encoder import fen_to_tensor
from src.models.small.vocab import policy_index, policy_to_idx

import random
import chess
import struct

# Fallback index when a move is not present in the vocabulary.
NULL_POLICY_INDEX = len(policy_index) - 1


def _policy_str_to_idx(move: str) -> int:
    if not move:
        return NULL_POLICY_INDEX
    move = move.strip()
    return policy_to_idx.get(move, NULL_POLICY_INDEX)


@dataclass
class SelfPlayBatch:
    planes: np.ndarray  # [B, 1, 8, 8, 19]
    policy_indices: np.ndarray  # [B]
    values: np.ndarray  # [B, 3]
    mask: np.ndarray  # [B] bool for valid value targets
    fens: List[str]


class SelfPlayDataset:
    """Random sampler over the Parquet buffer produced by self-play."""

    def __init__(self, path: Path, *, reload_seconds: float = 60.0) -> None:
        self.path = path
        self.reload_seconds = reload_seconds
        self._fens: List[str] = []
        self._values: List[tuple[float, float, float]] = []
        self._policies: List[str] = []
        self._size = 0
        self._last_load = 0.0
        self._last_mtime = 0.0
        self._rng = np.random.default_rng()

        self.reload(force=True)

    # ------------------------------------------------------------------ #
    def reload(self, *, force: bool = False) -> None:
        if not force:
            now = time.monotonic()
            if now - self._last_load < self.reload_seconds:
                return
            try:
                mtime = self.path.stat().st_mtime
            except FileNotFoundError:
                return
            if mtime <= self._last_mtime:
                self._last_load = now
                return

        # Retry loop for reading the file to handle race conditions
        max_retries = 50
        for attempt in range(max_retries):
            try:
                if self.path.suffix == ".parquet":
                    table = pq.read_table(self.path)
                    if "fen" not in table.column_names or "value" not in table.column_names:
                        raise RuntimeError(f"Parquet file {self.path} missing required columns.")

                    self._fens = table.column("fen").to_pylist()
                    raw_values = table.column("value").to_pylist()
                    self._values = [
                        _ensure_tuple(row) for row in raw_values  # type: ignore[arg-type]
                    ]
                    self._policies = table.column("policy").to_pylist()
                else:
                    # Assume binary format
                    self._fens, self._policies, self._values = self._read_binary_file()
                
                self._size = len(self._fens)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to reload dataset after {max_retries} attempts: {e}")
                    raise e
                time.sleep(0.1)

        self._last_load = time.monotonic()
        try:
            self._last_mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._last_mtime = self._last_load

    def _read_binary_file(self):
        fens = []
        policies = []
        values = []
        try:
            with self.path.open("rb") as f:
                content = f.read()
            
            offset = 0
            size = len(content)
            RECORD_SIZE = 128
            
            while offset + RECORD_SIZE <= size:
                # Read FEN (100 bytes)
                fen_bytes = content[offset:offset+100]
                try:
                    fen = fen_bytes.split(b'\0', 1)[0].decode("utf-8")
                except:
                    fen = ""
                offset += 100
                
                # Read Policy (16 bytes)
                pol_bytes = content[offset:offset+16]
                try:
                    policy = pol_bytes.split(b'\0', 1)[0].decode("utf-8")
                except:
                    policy = ""
                offset += 16
                
                # Read Value (12 bytes)
                val = struct.unpack_from("fff", content, offset)
                offset += 12
                
                if fen and policy:
                    fens.append(fen)
                    policies.append(policy)
                    values.append(val)
        except Exception as e:
            print(f"Error reading binary file: {e}")
            # Return partial results or empty
            return [], [], []
            
        return fens, policies, values

    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return self._size

    def sample_batch(self, batch_size: int) -> SelfPlayBatch:
        if self._size == 0:
            raise RuntimeError("Self-play dataset is empty. Run self-play generator first.")

        indices = self._rng.integers(0, self._size, size=batch_size, endpoint=False)
        planes = np.zeros((batch_size, 1, 8, 8, 19), dtype=np.float32)
        values = np.zeros((batch_size, 3), dtype=np.float32)
        masks = np.zeros((batch_size,), dtype=bool)
        policy_indices = np.full((batch_size,), NULL_POLICY_INDEX, dtype=np.int64)
        fens: List[str] = []

        for row, idx in enumerate(indices):
            fen = self._fens[idx]
            wdl = list(self._values[idx])
            policy_str = self._policies[idx]
            if _fen_white_to_move(fen):
                wdl = mirror_wdl(wdl)

            if random.random() < 0.5:
                fen = mirror_fen(fen)
                wdl = mirror_wdl(wdl)
                policy_str = mirror_policy_str(policy_str)
            fens.append(fen)
            planes[row, 0] = fen_to_tensor(fen)
            values[row] = wdl
            masks[row] = bool(np.sum(wdl) > 0)

            policy_indices[row] = _policy_str_to_idx(policy_str)

        return SelfPlayBatch(
            planes=planes,
            policy_indices=policy_indices,
            values=values,
            mask=masks,
            fens=fens,
        )


def _fen_white_to_move(fen: str) -> bool:
    try:
        parts = fen.split()
        return len(parts) > 1 and parts[1].lower() == "w"
    except Exception:
        return False


def _ensure_tuple(row) -> tuple[float, float, float]:
    if row is None:
        return (0.0, 0.0, 0.0)
    if isinstance(row, (list, tuple)):
        vals = list(row)
        vals += [0.0] * max(0, 3 - len(vals))
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    try:
        scalar = float(row)
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)
    return (scalar, 0.0, 0.0)

def mirror_wdl(wdl: tuple[float, float, float]) -> tuple[float, float, float]:
    return (wdl[2], wdl[1], wdl[0])

def mirror_fen(fen: str) -> str:
    board = chess.Board(fen)
    board.mirror()
    return board.fen()

def mirror_policy_str(policy_str: str) -> str:
    i,j = int(policy_str[1]),int(policy_str[3])

    return f"{policy_str[0]}{9-j}{policy_str[2]}{9-i}{policy_str[4:]}"