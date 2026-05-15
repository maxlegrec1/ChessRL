"""GPU expansion kernel == CPU reference, bit-for-bit.

The CPU reference (encode_position_packed_roundtrip) is already pinned to
the dense ground truth by test_packed_parity.py. This proves the CUDA
kernel reproduces that reference exactly (fp32 output for an exact
compare), so the runtime fp16 path = __float2half(reference) elementwise.
Load-bearing gate for the packed-bitboard refactor.
"""
from __future__ import annotations

import chess as pchess
import numpy as np
import pytest

import _inference_cpp

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

CASES = [
    ("startpos no history", STARTPOS, []),
    ("startpos + e4", STARTPOS, ["e2e4"]),
    ("kiwipete zero history", KIWIPETE, []),
    ("7-ply sicilian", STARTPOS,
     ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]),
    ("knight-dance repetition", STARTPOS,
     ["g1f3", "g8f6", "f3g1", "f6g8"]),
]


@pytest.mark.parametrize("name,fen,moves", CASES, ids=[c[0] for c in CASES])
def test_gpu_matches_cpu_reference(name, fen, moves):
    cpu = np.asarray(
        _inference_cpp.encode_position_packed_roundtrip(fen, moves),
        dtype=np.float32)
    gpu = np.asarray(
        _inference_cpp.expand_packed_gpu(fen, moves), dtype=np.float32)
    assert cpu.shape == gpu.shape
    assert np.array_equal(cpu, gpu), (
        f"{name}: first diffs "
        f"{np.argwhere(cpu != gpu)[:5].tolist()}")


def test_gpu_matches_cpu_random_games():
    rng = np.random.default_rng(20260515)
    bad = 0
    for g in range(30):
        board = pchess.Board()
        moves: list[str] = []
        for _ in range(int(rng.integers(0, 60))):
            legal = list(board.legal_moves)
            if not legal:
                break
            mv = legal[int(rng.integers(0, len(legal)))]
            moves.append(mv.uci())
            board.push(mv)
        cpu = np.asarray(
            _inference_cpp.encode_position_packed_roundtrip(STARTPOS, moves),
            dtype=np.float32)
        gpu = np.asarray(
            _inference_cpp.expand_packed_gpu(STARTPOS, moves), dtype=np.float32)
        if not np.array_equal(cpu, gpu):
            bad += 1
            print(f"game {g} ({len(moves)} ply): "
                  f"{np.argwhere(cpu != gpu)[:5].tolist()}")
    assert bad == 0, f"{bad}/30 random games mismatched GPU vs CPU"
