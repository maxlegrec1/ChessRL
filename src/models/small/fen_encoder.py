"""111-plane AlphaZero-style encoder.

Channel layout (kept absolute, not STM-canonicalized):

  Slot 0  (current position):
    [ 0..11]: 12 piece planes  (W:PNBRQK then B:PNBRQK)
    [12]   : repetition flag   (1.0 if this position seen before in game)
  Slots 1..7 (t-1 .. t-7), each is a 13-plane block of the same shape:
    [13..24]   t-1 piece planes;     [25]    t-1 repetition flag
    [26..37]   t-2 piece planes;     [38]    t-2 repetition flag
    [39..50]   t-3 piece planes;     [51]    t-3 repetition flag
    [52..63]   t-4 piece planes;     [64]    t-4 repetition flag
    [65..76]   t-5 piece planes;     [77]    t-5 repetition flag
    [78..89]   t-6 piece planes;     [90]    t-6 repetition flag
    [91..102]  t-7 piece planes;     [103]   t-7 repetition flag
  Global planes (broadcast over the 8x8 board):
    [104]: side-to-move (1.0 if white)
    [105]: en passant target square (one-hot at the EP square)
    [106]: castling W kingside
    [107]: castling W queenside
    [108]: castling B kingside
    [109]: castling B queenside
    [110]: halfmove clock / 100 (clamped to 1.0)

Total: 111 channels. Per-slot piece planes use a tensor row index of
`7 - rank` to put rank 1 at the bottom, matching the legacy convention
that the rest of the codebase (and the C++ encoder) follow.

When no history is provided, slots 1..7 are zero (all piece planes empty +
repetition flag 0). Live positions always contain pieces, so an all-zero
slot is itself an unambiguous "no-history" signal.
"""
from __future__ import annotations

import bulletchess
import numpy as np
import torch


INPUT_PLANES = 111
HISTORY_DEPTH = 8           # current + 7 past
_PAST_SLOTS = HISTORY_DEPTH - 1  # 7
_PIECE_PLANES = 12
_HIST_BLOCK = _PIECE_PLANES + 1  # 13 planes per slot

# Plane indices for the global section.
PLANE_STM = 104
PLANE_EP = 105
PLANE_CASTLING_WK = 106
PLANE_CASTLING_WQ = 107
PLANE_CASTLING_BK = 108
PLANE_CASTLING_BQ = 109
PLANE_HMC = 110

_TYPE_ORDER = (
    (bulletchess.PAWN, 0),
    (bulletchess.KNIGHT, 1),
    (bulletchess.BISHOP, 2),
    (bulletchess.ROOK, 3),
    (bulletchess.QUEEN, 4),
    (bulletchess.KING, 5),
)


def _slot_base(slot: int) -> int:
    """Channel index of the start of slot `slot` (0=current, 1..7=past)."""
    if slot == 0:
        return 0
    return _PIECE_PLANES + 1 + (slot - 1) * _HIST_BLOCK  # 13, 26, 39, ...


def _encode_piece_planes(board: "bulletchess.Board", tensor: np.ndarray, base: int) -> None:
    """Fill 12 piece planes starting at channel `base` from a bulletchess Board."""
    for color in (bulletchess.WHITE, bulletchess.BLACK):
        color_offset = 0 if color == bulletchess.WHITE else 6
        for piece_type, type_offset in _TYPE_ORDER:
            bitboard = board[color, piece_type]
            for sq in bitboard:
                idx = sq.index()
                rank = idx // 8
                file = idx % 8
                tensor[7 - rank, file, base + color_offset + type_offset] = 1.0


def _encode_globals(board: "bulletchess.Board", tensor: np.ndarray) -> None:
    """Fill the 7 global planes (104..110) from a bulletchess Board."""
    tensor[:, :, PLANE_STM] = 1.0 if board.turn == bulletchess.WHITE else 0.0
    if board.en_passant_square is not None:
        ep_idx = board.en_passant_square.index()
        rank = ep_idx // 8
        file = ep_idx % 8
        tensor[7 - rank, file, PLANE_EP] = 1.0
    tensor[:, :, PLANE_CASTLING_WK] = 1.0 if (bulletchess.WHITE_KINGSIDE  in board.castling_rights) else 0.0
    tensor[:, :, PLANE_CASTLING_WQ] = 1.0 if (bulletchess.WHITE_QUEENSIDE in board.castling_rights) else 0.0
    tensor[:, :, PLANE_CASTLING_BK] = 1.0 if (bulletchess.BLACK_KINGSIDE  in board.castling_rights) else 0.0
    tensor[:, :, PLANE_CASTLING_BQ] = 1.0 if (bulletchess.BLACK_QUEENSIDE in board.castling_rights) else 0.0
    tensor[:, :, PLANE_HMC] = min(board.halfmove_clock / 100.0, 1.0)


def fen_to_tensor(
    fen: str,
    past_boards: list["bulletchess.Board"] | None = None,
    past_repetitions: list[bool] | None = None,
    current_repetition: bool = False,
) -> np.ndarray:
    """Encode a position with optional history into [8, 8, 111] float32.

    Args:
        fen: current position as FEN string.
        past_boards: up to 7 bulletchess Boards, ordered newest-first
            (`past_boards[0]` is the position one ply ago). Shorter lists are
            zero-padded into later history slots. None or [] = no history.
        past_repetitions: parallel boolean list — `past_repetitions[i]` is
            True if `past_boards[i]` had already appeared earlier in the game.
            Computed by the caller from the full game history (chess library's
            `is_repetition`-style checks), since the encoder only sees the
            last 7 boards. Defaults to all False.
        current_repetition: True if the position described by `fen` had
            already appeared earlier in the game.
    """
    board = bulletchess.Board.from_fen(fen)
    tensor = np.zeros((8, 8, INPUT_PLANES), dtype=np.float32)

    # Slot 0: current piece planes + repetition.
    _encode_piece_planes(board, tensor, base=0)
    if current_repetition:
        tensor[:, :, _PIECE_PLANES] = 1.0  # plane 12

    # Slots 1..7: historical piece planes + per-slot repetition.
    past_boards = past_boards or []
    past_repetitions = past_repetitions or []
    for slot in range(1, HISTORY_DEPTH):
        past_idx = slot - 1
        if past_idx >= len(past_boards):
            break
        b = past_boards[past_idx]
        if b is None:
            continue
        base = _slot_base(slot)
        _encode_piece_planes(b, tensor, base=base)
        if past_idx < len(past_repetitions) and past_repetitions[past_idx]:
            tensor[:, :, base + _PIECE_PLANES] = 1.0

    # Globals (planes 104..110).
    _encode_globals(board, tensor)
    return tensor


def boards_to_tensor(
    boards,
    past_boards_per_position: list[list["bulletchess.Board"] | None] | None = None,
    past_repetitions_per_position: list[list[bool] | None] | None = None,
    current_repetitions: list[bool] | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Batch-encode current boards (+ optional per-position history) into [B, 8, 8, 111]."""
    if not boards:
        return torch.empty((0, 8, 8, INPUT_PLANES), device=device, dtype=dtype)
    B = len(boards)
    out = np.zeros((B, 8, 8, INPUT_PLANES), dtype=np.float32)

    for i, board in enumerate(boards):
        # Slot 0: current pieces + globals.
        _encode_piece_planes(board, out[i], base=0)
        _encode_globals(board, out[i])
        if current_repetitions is not None and i < len(current_repetitions) and current_repetitions[i]:
            out[i, :, :, _PIECE_PLANES] = 1.0
        # Historical slots.
        if past_boards_per_position is None or i >= len(past_boards_per_position):
            continue
        pbs = past_boards_per_position[i] or []
        prs = (past_repetitions_per_position[i] or []) if past_repetitions_per_position else []
        for slot in range(1, HISTORY_DEPTH):
            past_idx = slot - 1
            if past_idx >= len(pbs):
                break
            b = pbs[past_idx]
            if b is None:
                continue
            base = _slot_base(slot)
            _encode_piece_planes(b, out[i], base=base)
            if past_idx < len(prs) and prs[past_idx]:
                out[i, :, :, base + _PIECE_PLANES] = 1.0

    return torch.from_numpy(out).to(device=device, dtype=dtype)


def batch_fen_to_tensor(
    fens,
    past_boards_per_position: list[list["bulletchess.Board"] | None] | None = None,
    past_repetitions_per_position: list[list[bool] | None] | None = None,
    current_repetitions: list[bool] | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Batch FEN encoder. Equivalent to boards_to_tensor but takes FEN strings."""
    if not fens:
        return torch.empty((0, 8, 8, INPUT_PLANES), device=device, dtype=dtype)
    boards = [bulletchess.Board.from_fen(f) for f in fens]
    return boards_to_tensor(
        boards,
        past_boards_per_position=past_boards_per_position,
        past_repetitions_per_position=past_repetitions_per_position,
        current_repetitions=current_repetitions,
        device=device,
        dtype=dtype,
    )


def apply_move_to_tensor(
    board_tensor: torch.Tensor,
    move_uci: str,
    board: "bulletchess.Board",
) -> torch.Tensor:
    """Return a new tensor after applying `move_uci` to `board`.

    History note: this helper does NOT carry old history forward — it
    re-encodes from the post-move board only, so the result has all
    historical slots zeroed. Callers that need rolling history must
    construct the tensor themselves with `boards_to_tensor(..., past_boards_per_position=...)`.
    """
    mv = bulletchess.Move.from_uci(move_uci)
    board.apply(mv)
    try:
        new_tensor = boards_to_tensor(
            [board], device=board_tensor.device.type, dtype=board_tensor.dtype
        )
        return new_tensor[0]
    finally:
        board.undo()


# --------------------------------------------------------------------------- #
#                     Bitboard-based fast paths
# --------------------------------------------------------------------------- #
#
# The trainer's dataset reader stores 7 past board snapshots as raw uint64
# piece bitboards (12 per snapshot). Going through bulletchess.Board to
# decode them per record would be wasteful. These helpers fill a tensor
# directly from arrays of uint64.

def encode_piece_planes_from_bitboards(
    bitboards: np.ndarray,
    tensor: np.ndarray,
    base: int,
) -> None:
    """Fill 12 piece planes from a (12,) uint64 array.

    Bit-unpacking mirrors `cache_chessfens.fen_to_tensor_fast` (post-fix):
    byte 0 of a uint64 holds bits for rank 1; we flip both rank and file
    axes so the tensor matches the bulletchess loop convention
    (`tensor[7-rank, file, ...]`).
    """
    bbs = np.asarray(bitboards, dtype=np.uint64)
    if bbs.shape != (_PIECE_PLANES,):
        raise ValueError(f"expected 12 uint64 bitboards, got shape {bbs.shape}")
    bits = np.unpackbits(bbs.view(np.uint8)).reshape(_PIECE_PLANES, 8, 8)[:, ::-1, ::-1]
    tensor[:, :, base : base + _PIECE_PLANES] = np.transpose(bits.astype(np.float32), (1, 2, 0))


def encode_record_planes(
    current_fen: str,
    past_bitboards: np.ndarray,
    past_repetitions: np.ndarray,
    current_repetition: bool = False,
) -> np.ndarray:
    """Build a [8, 8, 111] tensor from buffer-record fields.

    Args:
        current_fen: FEN of the current position.
        past_bitboards: (7, 12) uint64 — piece bitboards for t-1..t-7
            (oldest first slot index 6, newest first slot index 0).
            Slots whose entries are all-zero are treated as "no history".
        past_repetitions: (7,) uint8 — repetition flag for t-1..t-7.
        current_repetition: repetition flag for the current position.
    """
    board = bulletchess.Board.from_fen(current_fen)
    tensor = np.zeros((8, 8, INPUT_PLANES), dtype=np.float32)

    _encode_piece_planes(board, tensor, base=0)
    if current_repetition:
        tensor[:, :, _PIECE_PLANES] = 1.0
    _encode_globals(board, tensor)

    pb = np.asarray(past_bitboards, dtype=np.uint64).reshape(_PAST_SLOTS, _PIECE_PLANES)
    pr = np.asarray(past_repetitions, dtype=np.uint8).reshape(_PAST_SLOTS)
    for slot in range(1, HISTORY_DEPTH):
        past_idx = slot - 1
        bb = pb[past_idx]
        if not bb.any():
            continue
        base = _slot_base(slot)
        encode_piece_planes_from_bitboards(bb, tensor, base=base)
        if pr[past_idx]:
            tensor[:, :, base + _PIECE_PLANES] = 1.0
    return tensor


def batch_encode_records(
    current_fens: list[str],
    past_bitboards_batch: np.ndarray,
    past_repetitions_batch: np.ndarray,
    current_repetitions: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized version of `encode_record_planes` over a batch.

    Args:
        current_fens: list[str] of length B.
        past_bitboards_batch: (B, 7, 12) uint64.
        past_repetitions_batch: (B, 7) uint8.
        current_repetitions: (B,) uint8 or None (treated as zeros).
    Returns:
        (B, 8, 8, 111) float32.
    """
    B = len(current_fens)
    out = np.zeros((B, 8, 8, INPUT_PLANES), dtype=np.float32)
    pb = np.asarray(past_bitboards_batch, dtype=np.uint64).reshape(B, _PAST_SLOTS, _PIECE_PLANES)
    pr = np.asarray(past_repetitions_batch, dtype=np.uint8).reshape(B, _PAST_SLOTS)
    cr = (np.asarray(current_repetitions, dtype=np.uint8).reshape(B)
          if current_repetitions is not None else np.zeros(B, dtype=np.uint8))
    for i, fen in enumerate(current_fens):
        board = bulletchess.Board.from_fen(fen)
        _encode_piece_planes(board, out[i], base=0)
        _encode_globals(board, out[i])
        if cr[i]:
            out[i, :, :, _PIECE_PLANES] = 1.0
        for slot in range(1, HISTORY_DEPTH):
            past_idx = slot - 1
            bb = pb[i, past_idx]
            if not bb.any():
                continue
            base = _slot_base(slot)
            encode_piece_planes_from_bitboards(bb, out[i], base=base)
            if pr[i, past_idx]:
                out[i, :, :, base + _PIECE_PLANES] = 1.0
    return out
