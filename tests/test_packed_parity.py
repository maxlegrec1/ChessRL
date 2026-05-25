"""Packed-path parity: pack(board,history) → CPU-expand MUST equal the
dense encoder bit-for-bit.

This is the load-bearing gate for the packed-bitboard refactor (same risk
class as the rank-flip bug). The dense encoder
(`encode_position_with_history`) is the ground truth pinned by
test_encoder_parity.py against the Python encoder; here we prove the
compact ~776 B representation + its CPU expansion reproduce it exactly.
The GPU kernel (Step 2) is then validated against this same CPU reference.
"""

from __future__ import annotations

import _inference_cpp
import chess as pchess
import numpy as np
import pytest

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

CASES = [
    ("startpos no history", STARTPOS, []),
    ("startpos + e4", STARTPOS, ["e2e4"]),
    ("kiwipete zero history", KIWIPETE, []),
    ("7-ply sicilian", STARTPOS, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]),
    ("knight-dance repetition", STARTPOS, ["g1f3", "g8f6", "f3g1", "f6g8"]),
    ("black to move", STARTPOS, ["e2e4"]),
]


@pytest.mark.parametrize("name,fen,moves", CASES, ids=[c[0] for c in CASES])
def test_packed_matches_dense(name, fen, moves):
    dense = np.asarray(_inference_cpp.encode_position_with_history(fen, moves), dtype=np.float32)
    packed = np.asarray(
        _inference_cpp.encode_position_packed_roundtrip(fen, moves), dtype=np.float32
    )
    assert dense.shape == packed.shape
    assert np.array_equal(dense, packed), (
        f"{name}: mismatch at {np.argwhere(dense != packed)[:5].tolist()}"
    )


def test_packed_random_games():
    """Many random games (varied castling/ep/halfmove/repetition/history)."""
    rng = np.random.default_rng(20260515)
    mismatches = 0
    for g in range(40):
        board = pchess.Board()
        moves: list[str] = []
        nply = int(rng.integers(0, 60))
        for _ in range(nply):
            legal = list(board.legal_moves)
            if not legal:
                break
            mv = legal[int(rng.integers(0, len(legal)))]
            moves.append(mv.uci())
            board.push(mv)
        dense = np.asarray(
            _inference_cpp.encode_position_with_history(STARTPOS, moves), dtype=np.float32
        )
        packed = np.asarray(
            _inference_cpp.encode_position_packed_roundtrip(STARTPOS, moves), dtype=np.float32
        )
        if not np.array_equal(dense, packed):
            mismatches += 1
            print(f"game {g} ({len(moves)} ply): {np.argwhere(dense != packed)[:5].tolist()}")
    assert mismatches == 0, f"{mismatches}/40 random games mismatched"
