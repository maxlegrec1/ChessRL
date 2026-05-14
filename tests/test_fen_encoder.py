import bulletchess
import numpy as np

from src.models.small.fen_encoder import (
    INPUT_PLANES,
    HISTORY_DEPTH,
    PLANE_STM,
    PLANE_HMC,
    fen_to_tensor,
    encode_record_planes,
    encode_piece_planes_from_bitboards,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_startpos_shape():
    t = fen_to_tensor(START_FEN)
    assert t.shape == (8, 8, INPUT_PLANES)
    assert t.shape == (8, 8, 111)


def test_white_to_move_plane():
    t = fen_to_tensor(START_FEN)
    assert (t[:, :, PLANE_STM] == 1.0).all()


def test_no_history_zero_planes():
    """When no past_boards are passed, slots 1..7 must be all zero."""
    t = fen_to_tensor(START_FEN)
    # Planes 13..103 are the seven 13-plane historical blocks. Must all be zero.
    assert t[:, :, 13:104].sum() == 0.0


def test_current_repetition_flag():
    """current_repetition=True lights up plane 12."""
    t = fen_to_tensor(START_FEN, current_repetition=True)
    assert (t[:, :, 12] == 1.0).all()
    # Other planes around it unaffected.
    t_no = fen_to_tensor(START_FEN, current_repetition=False)
    assert (t_no[:, :, 12] == 0.0).all()


def test_history_slot_filled():
    """A non-empty past_boards entry must populate the matching slot."""
    past = [bulletchess.Board.from_fen(START_FEN)]
    past_reps = [True]
    t = fen_to_tensor(
        START_FEN, past_boards=past, past_repetitions=past_reps
    )
    # Slot 1 piece planes: 13..24. Slot-1 repetition: 25.
    # Startpos has white pawns on rank 2 -> tensor row 6 for plane 13 (W:P).
    assert t[6, :, 13].sum() == 8.0  # 8 white pawns
    assert (t[:, :, 25] == 1.0).all()
    # Slots 2..7 still empty.
    assert t[:, :, 26:104].sum() == 0.0


def test_repetition_after_knight_dance():
    """Nf3 / Nf6 / Ng1 / Ng8 returns to the startpos with one prior occurrence."""
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"]
    board = bulletchess.Board.from_fen(START_FEN)
    seen_zobrist = {}
    past_boards = []
    past_reps = []

    def snapshot(b):
        # Use a piece-bitboards-based "hash" (color, type, square set) so we
        # don't depend on bulletchess exposing zobrist.
        h = tuple(
            int(b[c, p]) for c in (bulletchess.WHITE, bulletchess.BLACK)
            for p in (bulletchess.PAWN, bulletchess.KNIGHT, bulletchess.BISHOP,
                      bulletchess.ROOK, bulletchess.QUEEN, bulletchess.KING)
        ) + (b.turn == bulletchess.WHITE,)
        return h

    seen_zobrist[snapshot(board)] = 1
    for mv in moves:
        b_before = bulletchess.Board.from_fen(_board_to_fen(board))
        past_boards.insert(0, b_before)  # newest-first
        # Repetition flag for that past board: was it seen prior?
        past_reps.insert(0, seen_zobrist.get(snapshot(b_before), 0) > 1)
        board.apply(bulletchess.Move.from_uci(mv))
        key = snapshot(board)
        seen_zobrist[key] = seen_zobrist.get(key, 0) + 1

    # `board` is now the startpos again, so its current_repetition should be True.
    cur_rep = seen_zobrist[snapshot(board)] > 1
    t = fen_to_tensor(
        _board_to_fen(board),
        past_boards=past_boards[:7],
        past_repetitions=past_reps[:7],
        current_repetition=cur_rep,
    )
    assert (t[:, :, 12] == 1.0).all(), "current position has occurred before"


def _board_to_fen(b):
    # Minimal helper: bulletchess.Board has .fen()-ish accessor? Use built-in:
    return b.fen()


def test_encode_record_planes_roundtrip():
    """Building from FEN+bitboards equals building from FEN+bulletchess boards."""
    past = [bulletchess.Board.from_fen(START_FEN)] * 3
    past_reps_list = [False, True, False]
    t_from_boards = fen_to_tensor(
        START_FEN, past_boards=past, past_repetitions=past_reps_list
    )

    # Build the bitboards array.
    types = [bulletchess.PAWN, bulletchess.KNIGHT, bulletchess.BISHOP,
             bulletchess.ROOK, bulletchess.QUEEN, bulletchess.KING]
    bbs_per_slot = []
    for _ in range(3):
        bb = bulletchess.Board.from_fen(START_FEN)
        slot_bb = []
        for color in (bulletchess.WHITE, bulletchess.BLACK):
            for pt in types:
                slot_bb.append(int(bb[color, pt]))
        bbs_per_slot.append(slot_bb)
    pb = np.zeros((7, 12), dtype=np.uint64)
    pr = np.zeros(7, dtype=np.uint8)
    for i, bb in enumerate(bbs_per_slot):
        pb[i] = bb
    pr[1] = 1
    t_from_record = encode_record_planes(
        START_FEN, past_bitboards=pb, past_repetitions=pr
    )
    np.testing.assert_array_equal(t_from_boards, t_from_record)
