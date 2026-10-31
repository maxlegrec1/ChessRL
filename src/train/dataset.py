"""Replay buffer reader for v2 self-play records (see cpp/selfplay.cpp).

The buffer is a flat file of fixed 1280-byte records written by the C++
self-play worker. The trainer memory-maps it, samples rows (optionally from a
window of the newest records when the buffer is a FIFO ring), and decodes
input planes from the stored bitboards with tensor ops — no FEN parsing, no
python-chess, and the decode can run on the GPU.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.small.fen_encoder import INPUT_PLANES
from src.models.small.vocab import policy_index, policy_to_idx

RECORD_SIZE = 1280
RECORD_VERSION = 2
MAX_POLICY_ENTRIES = 84
VOCAB_SIZE = len(policy_index)
NO_MOVE = 0xFFFF  # played_idx sentinel in the record
IGNORE_INDEX = -100  # cross_entropy ignore_index for rows without a policy target

FLAG_FULL_SEARCH = 1
FLAG_SAMPLED_MOVE = 2
FLAG_RESIGNED = 4
FLAG_PLAYTHROUGH = 8
FLAG_CURRICULUM = 16
FLAG_MAX_PLIES = 32

RECORD_DTYPE = np.dtype(
    [
        ("fen", "S100"),
        ("played", "S16"),
        ("value_outcome", "<f4", (3,)),
        ("value_mcts", "<f4", (3,)),
        ("plies_remaining", "<f4"),
        ("backed_plies", "<f4"),
        ("past_rep", "u1", (7,)),
        ("cur_rep", "u1"),
        ("_pad", "V4"),
        ("past_bb", "<u8", (7, 12)),
        ("cur_bb", "<u8", (12,)),
        ("version", "u1"),
        ("flags", "u1"),
        ("castling", "u1"),
        ("ep", "u1"),
        ("halfmove", "<u2"),
        ("fullmove", "<u2"),
        ("ply", "<u2"),
        ("sims", "<u2"),
        ("n_policy", "<u2"),
        ("played_idx", "<u2"),
        ("policy", "<u2", (MAX_POLICY_ENTRIES, 2)),
    ]
)
assert RECORD_DTYPE.itemsize == RECORD_SIZE


def _mirror_move(move: str) -> str:
    # Vertical flip: ranks 1<->8, files unchanged (python-chess Board.mirror()).
    if len(move) < 4 or not move[1].isdigit() or not move[3].isdigit():
        return move
    return f"{move[0]}{9 - int(move[1])}{move[2]}{9 - int(move[3])}{move[4:]}"


# MIRROR_INDEX[i] = vocab index of move i after a colour flip (an involution).
MIRROR_INDEX = np.array(
    [policy_to_idx.get(_mirror_move(m), i) for i, m in enumerate(policy_index)], dtype=np.int64
)
_CASTLE_MOVES = {"e1g1": 0, "e1c1": 0, "e8g8": 1, "e8c8": 1}  # move -> colour of the king
CASTLE_INDEX = {policy_to_idx[m]: c for m, c in _CASTLE_MOVES.items()}


def window_size(total_writes: int, c: int, alpha: float, beta: float) -> int:
    """KataGo's growing replay window over the newest records."""
    if c <= 0:
        return 0
    n = max(total_writes, 1)
    if n <= c:
        return n
    return int(c * (1.0 + beta * ((n / c) ** alpha - 1.0) / alpha))


@dataclass
class Batch:
    """Raw fields of sampled records, as torch tensors (CPU until `.to()`)."""

    cur_bb: torch.Tensor  # [B, 12] int64 (uint64 bit patterns)
    past_bb: torch.Tensor  # [B, 7, 12] int64
    cur_rep: torch.Tensor  # [B] uint8
    past_rep: torch.Tensor  # [B, 7] uint8
    white_to_move: torch.Tensor  # [B] bool
    castling: torch.Tensor  # [B] uint8
    ep: torch.Tensor  # [B] int64, 255 = none
    halfmove: torch.Tensor  # [B] float32
    flags: torch.Tensor  # [B] uint8
    played_idx: torch.Tensor  # [B] int64, IGNORE_INDEX if none
    policy_idx: torch.Tensor  # [B, K] int64
    policy_prob: torch.Tensor  # [B, K] float32, zero for unused entries
    value_outcome: torch.Tensor  # [B, 3]
    value_mcts: torch.Tensor  # [B, 3]
    plies_remaining: torch.Tensor  # [B]
    backed_plies: torch.Tensor  # [B]
    mirror: torch.Tensor  # [B] bool, colour-flip augmentation applied at decode
    fens: list[str]

    def to(self, device: torch.device | str, non_blocking: bool = True) -> Batch:
        moved = {}
        for name, value in self.__dict__.items():
            moved[name] = (
                value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
            )
        return Batch(**moved)


def records_to_batch(
    rows: np.ndarray, mirror_prob: float = 0.0, rng: random.Random | None = None
) -> Batch:
    """Pack structured record rows into a Batch (no plane decoding yet)."""
    rng = rng or random
    b = len(rows)
    n_policy = np.minimum(rows["n_policy"].astype(np.int64), MAX_POLICY_ENTRIES)
    entry_ok = np.arange(MAX_POLICY_ENTRIES)[None, :] < n_policy[:, None]
    pol_idx = rows["policy"][:, :, 0].astype(np.int64)
    pol_prob = rows["policy"][:, :, 1].astype(np.float32) * entry_ok
    pol_idx = np.where(entry_ok, pol_idx, 0)
    played = rows["played_idx"].astype(np.int64)
    played = np.where(played == NO_MOVE, IGNORE_INDEX, played)
    mirror = np.array([rng.random() < mirror_prob for _ in range(b)], dtype=bool)

    # Colour flip of the targets happens here (index remap); planes flip at decode.
    if mirror.any():
        pol_idx = np.where(mirror[:, None], MIRROR_INDEX[pol_idx], pol_idx)
        ok = played >= 0
        played = np.where(mirror & ok, MIRROR_INDEX[np.where(ok, played, 0)], played)

    def t(a, dtype=None):
        out = torch.from_numpy(np.ascontiguousarray(a))
        return out if dtype is None else out.to(dtype)

    return Batch(
        cur_bb=t(rows["cur_bb"].view(np.int64)),
        past_bb=t(rows["past_bb"].view(np.int64)),
        cur_rep=t(rows["cur_rep"]),
        past_rep=t(rows["past_rep"]),
        white_to_move=t(np.char.find(rows["fen"], b" w ") >= 0),
        castling=t(rows["castling"]),
        ep=t(rows["ep"].astype(np.int64)),
        halfmove=t(rows["halfmove"].astype(np.float32)),
        flags=t(rows["flags"]),
        played_idx=t(played),
        policy_idx=t(pol_idx),
        policy_prob=t(pol_prob),
        value_outcome=t(rows["value_outcome"]),
        value_mcts=t(rows["value_mcts"]),
        plies_remaining=t(rows["plies_remaining"]),
        backed_plies=t(rows["backed_plies"]),
        mirror=t(mirror),
        fens=[f.decode("ascii", "replace") for f in rows["fen"]],
    )


_SHIFTS = torch.arange(64, dtype=torch.int64)
_SWAP_COLOURS = torch.tensor([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])


def decode_planes(batch: Batch, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build the [B, 8, 8, 111] input (same layout as fen_encoder.py) on batch's device."""
    return decode_fields(
        batch.cur_bb, batch.past_bb, batch.cur_rep, batch.past_rep, batch.white_to_move,
        batch.castling, batch.ep, batch.halfmove, batch.mirror, dtype,
    )  # fmt: skip


def decode_fields(cur_bb, past_bb, cur_rep, past_rep, white_to_move, castling, ep, halfmove, mirror,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:  # fmt: skip
    """decode_planes on the fields themselves. No host sync and no data-dependent shape, so the
    trainer compiles it with the network."""
    device = cur_bb.device
    bsz = cur_bb.shape[0]
    shifts = _SHIFTS.to(device)

    bb = torch.cat([cur_bb[:, None, :], past_bb], dim=1)  # [B, 8, 12]
    # bit k of each bitboard = square k = rank * 8 + file. Arithmetic shifts are
    # fine: & 1 keeps only the shifted bit even when bit 63 sign-extends.
    bits = (bb[..., None] >> shifts) & 1  # [B, 8, 12, 64]
    bits = bits.view(bsz, 8, 12, 8, 8)  # [..., rank, file]

    flipped = bits[:, :, _SWAP_COLOURS.to(device)].flip(-2)  # swap colours + rank flip
    bits = torch.where(mirror[:, None, None, None, None], flipped, bits)
    bits = bits.flip(-2)  # tensor row = 7 - rank (rank 1 at the bottom)

    reps = torch.cat([cur_rep[:, None], past_rep], dim=1).to(bits.dtype)  # [B, 8]
    hist = torch.cat(
        [bits, reps[:, :, None, None, None].expand(bsz, 8, 1, 8, 8)], dim=2
    )  # [B, 8 slots, 13, 8, 8]; channel = 13 * slot + k
    planes = torch.zeros(bsz, INPUT_PLANES, 8, 8, dtype=dtype, device=device)
    planes[:, :104] = hist.reshape(bsz, 104, 8, 8).to(dtype)

    wtm = white_to_move ^ mirror
    planes[:, 104] = wtm.to(dtype)[:, None, None]

    has_ep = ep < 64
    ep_sq = torch.where(has_ep, ep, torch.zeros_like(ep))
    ep_rank = ep_sq // 8
    ep_rank = torch.where(mirror, 7 - ep_rank, ep_rank)
    ep_file = ep_sq % 8
    ep_cell = F.one_hot((7 - ep_rank) * 8 + ep_file, 64).view(bsz, 8, 8)  # row 7 - rank, as the pieces
    planes[:, 105] = (ep_cell * has_ep[:, None, None]).to(dtype)

    cr = castling.to(torch.int64)
    wk, wq, bk, bq = (cr & 1) > 0, (cr & 2) > 0, (cr & 4) > 0, (cr & 8) > 0
    wk, bk = torch.where(mirror, bk, wk), torch.where(mirror, wk, bk)
    wq, bq = torch.where(mirror, bq, wq), torch.where(mirror, wq, bq)
    for ch, flag in ((106, wk), (107, wq), (108, bk), (109, bq)):
        planes[:, ch] = flag.to(dtype)[:, None, None]
    planes[:, 110] = (halfmove / 100.0).clamp(max=1.0).to(dtype)[:, None, None]
    return planes.permute(0, 2, 3, 1).contiguous()


def dense_policy_target(batch: Batch, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """[B, vocab] visit distribution (renormalized; rows without entries are all zero)."""
    return dense_target(batch.policy_idx, batch.policy_prob, vocab_size)


def dense_target(policy_idx, policy_prob, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    target = torch.zeros(policy_idx.shape[0], vocab_size, device=policy_idx.device)
    target.scatter_add_(1, policy_idx, policy_prob)
    total = target.sum(dim=1, keepdim=True)
    return target / total.clamp_min(1e-9)


class ReplayBuffer:
    """Memory-mapped view of the self-play buffer with optional recent-window sampling."""

    def __init__(
        self,
        path: Path,
        *,
        window_c: int = 0,
        window_alpha: float = 0.75,
        window_beta: float = 0.4,
        seed: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.writes_path = self.path.parent / (self.path.name + ".writes")
        self.stats_path = self.path.parent / (self.path.name + ".stats.json")
        self.window_c = window_c
        self.window_alpha = window_alpha
        self.window_beta = window_beta
        self._mm: np.ndarray | None = None
        self._mapped_records = 0
        self.total_writes = 0
        self.capacity = 0
        self.fifo = False
        self._rng = np.random.default_rng(seed)
        self.refresh()

    @property
    def size(self) -> int:
        return self._mapped_records

    def refresh(self) -> None:
        """Re-read the write counter and remap if the file grew."""
        try:
            n = self.path.stat().st_size // RECORD_SIZE
        except FileNotFoundError:
            n = 0
        if n != self._mapped_records:
            self._mm = np.memmap(self.path, dtype=RECORD_DTYPE, mode="r", shape=(n,)) if n else None
            self._mapped_records = n
        try:
            self.total_writes = int(self.writes_path.read_text().strip() or 0)
        except (FileNotFoundError, ValueError):
            self.total_writes = n
        try:
            meta = json.loads(self.stats_path.read_text())
            self.capacity = int(meta.get("capacity", n))
            self.fifo = bool(meta.get("fifo", False))
        except (FileNotFoundError, ValueError):
            self.capacity = max(self.capacity, n)

    def current_window(self) -> int:
        n = self._mapped_records
        if not self.fifo or self.window_c <= 0:
            return n
        w = window_size(self.total_writes, self.window_c, self.window_alpha, self.window_beta)
        return max(1, min(w, n, self.total_writes))

    def sample_indices(self, batch_size: int) -> np.ndarray:
        n = self._mapped_records
        if n == 0:
            raise RuntimeError("Replay buffer is empty. Run self-play first.")
        w = self.current_window()
        if w >= n:
            return self._rng.integers(0, n, size=batch_size)
        # Newest record is at (total_writes - 1) % capacity; walk back r < w.
        r = self._rng.integers(0, w, size=batch_size)
        cap = max(self.capacity, n)
        idx = (self.total_writes - 1 - r) % cap
        return np.minimum(idx, n - 1)

    def sample_rows(self, batch_size: int) -> np.ndarray:
        assert self._mm is not None
        idx = np.sort(self.sample_indices(batch_size))  # sorted = friendlier page access
        rows = np.asarray(self._mm[idx])
        ok = rows["version"] == RECORD_VERSION
        if not ok.all():
            # Torn or foreign rows (a write racing the read): resample them.
            bad = int((~ok).sum())
            rows = (
                np.concatenate([rows[ok], self.sample_rows(bad)])
                if ok.any()
                else self.sample_rows(batch_size)
            )
        return rows

    def sample_batch(self, batch_size: int, mirror_prob: float = 0.5) -> Batch:
        return records_to_batch(self.sample_rows(batch_size), mirror_prob=mirror_prob)


class Prefetcher:
    """Background thread that keeps a few pinned CPU batches ready."""

    def __init__(self, buffer: ReplayBuffer, batch_size: int, mirror_prob: float, depth: int = 3):
        import queue
        import threading

        self.buffer = buffer
        self.batch_size = batch_size
        self.mirror_prob = mirror_prob
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    batch = self.buffer.sample_batch(self.batch_size, self.mirror_prob)
                pinned = {
                    k: (v.pin_memory() if torch.is_tensor(v) and torch.cuda.is_available() else v)
                    for k, v in batch.__dict__.items()
                }
                self._queue.put(Batch(**pinned))
            except Exception as exc:  # keep the trainer alive; surface on next get()
                self._queue.put(exc)
                time.sleep(1.0)

    def refresh(self) -> None:
        with self._lock:
            self.buffer.refresh()

    def get(self) -> Batch:
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()
