"""Cross-language parity for the 111-plane encoder.

The Python encoder (`src.models.small.fen_encoder.fen_to_tensor`) and the
C++ encoder (`encodeSmallFeatures` exposed via the new pybind hook
`encode_position_with_history`) MUST agree bit-for-bit on the input
tensor for every position the model ever sees. This was the load-bearing
verification when planning the AlphaZero history extension — same risk
class as the cache rank-flip bug we hit earlier.
"""

from __future__ import annotations

import _inference_cpp
import bulletchess
import chess as pchess
import numpy as np

from src.models.small.fen_encoder import INPUT_PLANES, fen_to_tensor

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def _encode_cpp(fen: str, moves: list[str]) -> np.ndarray:
    """C++ encoder → (8, 8, 111) float32."""
    flat = np.asarray(_inference_cpp.encode_position_with_history(fen, moves), dtype=np.float32)
    assert flat.size == 8 * 8 * INPUT_PLANES, flat.size
    return flat.reshape(8, 8, INPUT_PLANES)


def _encode_python(starting_fen: str, moves: list[str]) -> np.ndarray:
    """Python encoder → (8, 8, 111) float32.

    Replays `moves` from `starting_fen` to get the current board, plus the
    7 most recent past boards (newest first) and their per-slot repetition
    flags. Uses python-chess for is_repetition() because it tracks the full
    history; bulletchess is used only as the encoder backend so the planes
    match the live runtime convention.
    """
    pboard = pchess.Board(starting_fen)
    boards_chronological = [pboard.copy()]
    for mv in moves:
        pboard.push_uci(mv)
        boards_chronological.append(pboard.copy())

    # Repetition for each position is its is_repetition() at that point in time.
    rep_flags_chronological = []
    replay = pchess.Board(starting_fen)
    rep_flags_chronological.append(replay.is_repetition(2))  # 3-fold ⇒ count=2
    for mv in moves:
        replay.push_uci(mv)
        rep_flags_chronological.append(replay.is_repetition(2))

    past_fens = [b.fen() for b in boards_chronological[:-1]][::-1]  # newest first
    past_boards_bc = [bulletchess.Board.from_fen(f) for f in past_fens][:7]
    past_reps = [bool(r) for r in rep_flags_chronological[:-1]][::-1][:7]
    current_repetition = bool(rep_flags_chronological[-1])

    return fen_to_tensor(
        boards_chronological[-1].fen(),
        past_boards=past_boards_bc,
        past_repetitions=past_reps,
        current_repetition=current_repetition,
    )


def _diff_report(py: np.ndarray, cpp: np.ndarray) -> str:
    diffs = np.where(~np.isclose(py, cpp))
    if len(diffs[0]) == 0:
        return "no differences"
    planes = sorted(set(diffs[2].tolist()))
    return f"differing planes: {planes}  (first diff at {(diffs[0][0], diffs[1][0], diffs[2][0])} py={py[diffs[0][0], diffs[1][0], diffs[2][0]]}, cpp={cpp[diffs[0][0], diffs[1][0], diffs[2][0]]})"


def test_parity_startpos_no_history():
    moves: list[str] = []
    py = _encode_python(STARTPOS, moves)
    cpp = _encode_cpp(STARTPOS, moves)
    assert py.shape == cpp.shape == (8, 8, 111)
    assert np.array_equal(py, cpp), _diff_report(py, cpp)


def test_parity_startpos_plus_e4():
    moves = ["e2e4"]
    py = _encode_python(STARTPOS, moves)
    cpp = _encode_cpp(STARTPOS, moves)
    assert np.array_equal(py, cpp), _diff_report(py, cpp)


def test_parity_kiwipete_zero_history():
    py = _encode_python(KIWIPETE, [])
    cpp = _encode_cpp(KIWIPETE, [])
    assert np.array_equal(py, cpp), _diff_report(py, cpp)


def test_parity_seven_ply_history():
    moves = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]
    py = _encode_python(STARTPOS, moves)
    cpp = _encode_cpp(STARTPOS, moves)
    assert py.shape == (8, 8, 111)
    assert np.array_equal(py, cpp), _diff_report(py, cpp)


def test_parity_repetition_after_knight_dance():
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"]
    py = _encode_python(STARTPOS, moves)
    cpp = _encode_cpp(STARTPOS, moves)
    # Current position is the startpos again — current-repetition plane must
    # be lit in BOTH encoders. Plane 12 == current-rep flag.
    assert (py[:, :, 12] == 1.0).all()
    assert (cpp[:, :, 12] == 1.0).all()
    assert np.array_equal(py, cpp), _diff_report(py, cpp)


def test_parity_black_to_move():
    # Black-to-move position, fresh — STM plane must be 0 in both encoders,
    # piece planes in absolute frame (no canonicalization).
    moves = ["e2e4"]  # after this, black to move.
    py = _encode_python(STARTPOS, moves)
    cpp = _encode_cpp(STARTPOS, moves)
    assert (py[:, :, 104] == 0.0).all()
    assert (cpp[:, :, 104] == 0.0).all()
    assert np.array_equal(py, cpp), _diff_report(py, cpp)
