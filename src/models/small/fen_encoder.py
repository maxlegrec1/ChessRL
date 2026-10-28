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
`7 - rank` to put rank 1 at the bottom, the convention the rest of the
codebase (and the C++ encoder) follow.

When no history is provided, slots 1..7 are zero (all piece planes empty +
repetition flag 0). Live positions always contain pieces, so an all-zero
slot is itself an unambiguous "no-history" signal.
"""

from __future__ import annotations

import bulletchess
import numpy as np

INPUT_PLANES = 111
HISTORY_DEPTH = 8  # current + 7 past
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


def _encode_piece_planes(board: bulletchess.Board, tensor: np.ndarray, base: int) -> None:
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


def _encode_globals(board: bulletchess.Board, tensor: np.ndarray) -> None:
    """Fill the 7 global planes (104..110) from a bulletchess Board."""
    tensor[:, :, PLANE_STM] = 1.0 if board.turn == bulletchess.WHITE else 0.0
    if board.en_passant_square is not None:
        ep_idx = board.en_passant_square.index()
        rank = ep_idx // 8
        file = ep_idx % 8
        tensor[7 - rank, file, PLANE_EP] = 1.0
    tensor[:, :, PLANE_CASTLING_WK] = (
        1.0 if (bulletchess.WHITE_KINGSIDE in board.castling_rights) else 0.0
    )
    tensor[:, :, PLANE_CASTLING_WQ] = (
        1.0 if (bulletchess.WHITE_QUEENSIDE in board.castling_rights) else 0.0
    )
    tensor[:, :, PLANE_CASTLING_BK] = (
        1.0 if (bulletchess.BLACK_KINGSIDE in board.castling_rights) else 0.0
    )
    tensor[:, :, PLANE_CASTLING_BQ] = (
        1.0 if (bulletchess.BLACK_QUEENSIDE in board.castling_rights) else 0.0
    )
    tensor[:, :, PLANE_HMC] = min(board.halfmove_clock / 100.0, 1.0)


def fen_to_tensor(
    fen: str,
    past_boards: list[bulletchess.Board] | None = None,
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


# --------------------------------------------------------------------------- #
#                     Buffer records
# --------------------------------------------------------------------------- #
#
# Records store the 7 past positions as raw uint64 piece bitboards (12 per
# position). This reference encoder (used by the tests and
# scripts/check_buffer.py) fills the planes straight from those bitboards; the
# trainer decodes records in bulk in src/train/dataset.py.


def encode_piece_planes_from_bitboards(
    bitboards: np.ndarray,
    tensor: np.ndarray,
    base: int,
) -> None:
    """Fill 12 piece planes from a (12,) uint64 array.

    Byte 0 of each little-endian uint64 holds rank 1; both the rank and file
    axes are flipped after unpacking so the tensor matches the bulletchess
    loop convention (`tensor[7-rank, file, ...]`).
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
