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

# On-disk record size (bytes). Mirrors cpp/selfplay.cpp's kRecordSize.
RECORD_SIZE = 148


def _policy_str_to_idx(move: str) -> int:
    if not move:
        return NULL_POLICY_INDEX
    move = move.strip()
    # Normalize knight promotion suffixes (e.g., e7e8n) to align with vocab
    if move.endswith("n"):
        move = move[:-1]
    return policy_to_idx.get(move, NULL_POLICY_INDEX)


@dataclass
class SelfPlayBatch:
    planes: np.ndarray            # [B, 8, 8, 19]
    policy_indices: np.ndarray    # [B]
    value_outcomes: np.ndarray    # [B, 3] sparse one-hot (game outcome from side to move)
    value_mcts: np.ndarray        # [B, 3] dense WDL backed up from MCTS root
    plies_remaining: np.ndarray   # [B] float32 plies until game end; -1 = no target
    backed_plies: np.ndarray      # [B] float32 MCTS-backed plies-remaining at root; -1 = no target
    fens: List[str]


class SelfPlayDataset:
    """Random sampler over the Parquet buffer produced by self-play."""

    def __init__(self, path: Path, *, reload_seconds: float = 60.0) -> None:
        self.path = path
        self.reload_seconds = reload_seconds
        self._fens: List[str] = []
        self._value_outcomes: List[tuple[float, float, float]] = []
        self._value_mcts: List[tuple[float, float, float]] = []
        self._plies_remaining: List[float] = []
        self._backed_plies: List[float] = []
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
                    self._value_outcomes = [_ensure_tuple(row) for row in raw_values]
                    # Parquet path predates the dual-target schema; supply zero
                    # value_mcts so the value_q head sees a no-op target.
                    self._value_mcts = [(0.0, 0.0, 0.0)] * len(self._fens)
                    self._plies_remaining = [-1.0] * len(self._fens)
                    self._backed_plies = [-1.0] * len(self._fens)
                    self._policies = table.column("policy").to_pylist()
                else:
                    # Assume binary format
                    (
                        self._fens,
                        self._policies,
                        self._value_outcomes,
                        self._value_mcts,
                        self._plies_remaining,
                        self._backed_plies,
                    ) = self._read_binary_file()
                
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
        # On-disk record layout (148 bytes, fixed). Mirrors cpp/selfplay.cpp.
        FEN_OFFSET = 0
        POLICY_OFFSET = 100
        VALUE_OUTCOME_OFFSET = 116
        VALUE_MCTS_OFFSET = 128
        PLIES_REMAINING_OFFSET = 140
        BACKED_PLIES_OFFSET = 144

        fens: List[str] = []
        policies: List[str] = []
        value_outcomes: List[tuple[float, float, float]] = []
        value_mcts: List[tuple[float, float, float]] = []
        plies_remaining: List[float] = []
        backed_plies: List[float] = []
        try:
            with self.path.open("rb") as f:
                content = f.read()

            size = len(content)
            offset = 0
            while offset + RECORD_SIZE <= size:
                fen_bytes = content[offset + FEN_OFFSET : offset + POLICY_OFFSET]
                pol_bytes = content[offset + POLICY_OFFSET : offset + VALUE_OUTCOME_OFFSET]
                try:
                    fen = fen_bytes.split(b"\0", 1)[0].decode("utf-8")
                    policy = pol_bytes.split(b"\0", 1)[0].decode("utf-8")
                except Exception:
                    fen = policy = ""

                val_out = struct.unpack_from("fff", content, offset + VALUE_OUTCOME_OFFSET)
                val_mcts = struct.unpack_from("fff", content, offset + VALUE_MCTS_OFFSET)
                (plies,) = struct.unpack_from("f", content, offset + PLIES_REMAINING_OFFSET)
                (b_plies,) = struct.unpack_from("f", content, offset + BACKED_PLIES_OFFSET)
                offset += RECORD_SIZE

                if fen and policy:
                    fens.append(fen)
                    policies.append(policy)
                    value_outcomes.append(val_out)
                    value_mcts.append(val_mcts)
                    plies_remaining.append(float(plies))
                    backed_plies.append(float(b_plies))
        except Exception as e:
            print(f"Error reading binary file: {e}")
            return [], [], [], [], [], []

        return fens, policies, value_outcomes, value_mcts, plies_remaining, backed_plies

    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return self._size

    def sample_batch(self, batch_size: int) -> SelfPlayBatch:
        if self._size == 0:
            raise RuntimeError("Self-play dataset is empty. Run self-play generator first.")

        indices = self._rng.integers(0, self._size, size=batch_size, endpoint=False)
        planes = np.zeros((batch_size, 8, 8, 19), dtype=np.float32)
        value_outcomes = np.zeros((batch_size, 3), dtype=np.float32)
        value_mcts = np.zeros((batch_size, 3), dtype=np.float32)
        plies_remaining = np.full((batch_size,), -1.0, dtype=np.float32)
        backed_plies = np.full((batch_size,), -1.0, dtype=np.float32)
        policy_indices = np.full((batch_size,), NULL_POLICY_INDEX, dtype=np.int64)
        fens: List[str] = []

        for row, idx in enumerate(indices):
            fen = self._fens[idx]
            wdl_outcome = list(self._value_outcomes[idx])
            wdl_mcts = list(self._value_mcts[idx])
            plies = self._plies_remaining[idx] if idx < len(self._plies_remaining) else -1.0
            b_plies = self._backed_plies[idx] if idx < len(self._backed_plies) else -1.0
            policy_str = self._policies[idx]
            # Both WDLs are already in side-to-move perspective; mirroring the board
            # swaps colors AND swaps the side to move, so the WDL stays valid as-is.
            if random.random() < 0.5:
                fen = mirror_fen(fen)
                policy_str = mirror_policy_str(policy_str)
            fens.append(fen)
            planes[row] = fen_to_tensor(fen)
            value_outcomes[row] = wdl_outcome
            value_mcts[row] = wdl_mcts
            plies_remaining[row] = plies
            backed_plies[row] = b_plies
            policy_indices[row] = _policy_str_to_idx(policy_str)

        return SelfPlayBatch(
            planes=planes,
            policy_indices=policy_indices,
            value_outcomes=value_outcomes,
            value_mcts=value_mcts,
            plies_remaining=plies_remaining,
            backed_plies=backed_plies,
            fens=fens,
        )


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

def mirror_fen(fen: str) -> str:
    board = chess.Board(fen)
    mirrored = board.mirror()  # mirror() returns a NEW board, does not modify in place
    return mirrored.fen()

def mirror_policy_str(policy_str: str) -> str:
    i,j = int(policy_str[1]),int(policy_str[3])

    return f"{policy_str[0]}{9-j}{policy_str[2]}{9-i}{policy_str[4:]}"