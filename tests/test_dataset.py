"""v2 replay buffer: record layout, vectorized plane decode, colour flip, window sampling.

Records are synthesized in Python from random python-chess games with the same
byte layout cpp/selfplay.cpp writes; decoded planes must match the reference
per-record encoder (fen_encoder.encode_record_planes) exactly.
"""

import random

import chess
import numpy as np
import pytest
import torch

from src.models.small.fen_encoder import encode_record_planes
from src.models.small.vocab import policy_to_idx
from src.train.dataset import (
    FLAG_FULL_SEARCH,
    IGNORE_INDEX,
    MIRROR_INDEX,
    RECORD_DTYPE,
    RECORD_SIZE,
    ReplayBuffer,
    decode_planes,
    dense_policy_target,
    records_to_batch,
    window_size,
)

_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def bitboards(board: chess.Board) -> np.ndarray:
    out = np.zeros(12, dtype=np.uint64)
    for c, colour in enumerate((chess.WHITE, chess.BLACK)):
        for t, ptype in enumerate(_TYPES):
            out[c * 6 + t] = np.uint64(int(board.pieces_mask(ptype, colour)))
    return out


def vocab_key(move: chess.Move) -> str:
    uci = move.uci()
    return uci[:-1] if uci.endswith("n") else uci


def synth_records(n_games: int = 12, seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_games):
        board = chess.Board()
        past: list[np.ndarray] = []
        for ply in range(rng.randint(20, 90)):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            rec = np.zeros((), dtype=RECORD_DTYPE)
            fen = board.fen()
            rec["fen"] = fen.encode()
            move = rng.choice(legal)
            rec["played"] = move.uci().encode()
            rec["value_outcome"] = [0, 1, 0]
            rec["value_mcts"] = [0.2, 0.5, 0.3]
            rec["plies_remaining"] = 10
            rec["backed_plies"] = 12
            for k, bb in enumerate(past[:7]):
                rec["past_bb"][k] = bb
                rec["past_rep"][k] = rng.random() < 0.1
            rec["cur_rep"] = rng.random() < 0.1
            rec["cur_bb"] = bitboards(board)
            rec["version"] = 2
            rec["flags"] = FLAG_FULL_SEARCH
            cr = board.castling_rights
            rec["castling"] = (
                bool(cr & chess.BB_H1) * 1
                | bool(cr & chess.BB_A1) * 2
                | bool(cr & chess.BB_H8) * 4
                | bool(cr & chess.BB_A8) * 8
            )
            ep_field = fen.split()[3]
            rec["ep"] = 255 if ep_field == "-" else chess.parse_square(ep_field)
            rec["halfmove"] = board.halfmove_clock
            rec["fullmove"] = board.fullmove_number
            rec["ply"] = ply
            # A few-move visit distribution, played move first.
            others = rng.sample(legal, min(len(legal), 4))
            entries = [move] + [m for m in others if m != move]
            probs = np.array([rng.random() + 0.5 for _ in entries])
            probs /= probs.sum()
            rec["n_policy"] = len(entries)
            for k, (m, p) in enumerate(zip(entries, probs, strict=True)):
                rec["policy"][k] = (policy_to_idx[vocab_key(m)], int(round(p * 65535)))
            rec["played_idx"] = policy_to_idx[vocab_key(move)]
            rows.append(rec)
            past.insert(0, bitboards(board))
            board.push(move)
    return np.array(rows, dtype=RECORD_DTYPE)


def reference_planes(rec: np.void, mirror: bool) -> np.ndarray:
    fen = rec["fen"].decode()
    past = rec["past_bb"].copy()
    if mirror:
        fen = chess.Board(fen).mirror().fen()
        flipped = past.view(np.uint8).reshape(7, 12, 8)[:, :, ::-1].copy().view(np.uint64).reshape(7, 12)
        past = np.concatenate([flipped[:, 6:], flipped[:, :6]], axis=1)
    return encode_record_planes(
        fen, past_bitboards=past, past_repetitions=rec["past_rep"], current_repetition=bool(rec["cur_rep"])
    )


def test_record_dtype_is_1280_bytes():
    assert RECORD_DTYPE.itemsize == RECORD_SIZE == 1280


@pytest.mark.parametrize("mirror_prob", [0.0, 1.0, 0.5])
def test_decode_matches_reference_encoder(mirror_prob):
    rows = synth_records()
    batch = records_to_batch(rows, mirror_prob=mirror_prob, rng=random.Random(1))
    planes = decode_planes(batch).numpy()
    for i, rec in enumerate(rows):
        ref = reference_planes(rec, bool(batch.mirror[i]))
        np.testing.assert_array_equal(planes[i], ref, err_msg=f"row {i} fen={rec['fen']}")


def test_mirror_remaps_policy_targets():
    rows = synth_records(n_games=4, seed=3)
    plain = records_to_batch(rows, mirror_prob=0.0)
    flipped = records_to_batch(rows, mirror_prob=1.0)
    k = plain.policy_prob > 0
    assert torch.equal(flipped.policy_idx[k], torch.from_numpy(MIRROR_INDEX)[plain.policy_idx[k]])
    ok = plain.played_idx != IGNORE_INDEX
    assert torch.equal(flipped.played_idx[ok], torch.from_numpy(MIRROR_INDEX)[plain.played_idx[ok]])
    # Mirrored targets are legal in the mirrored position.
    for i, rec in enumerate(rows):
        board = chess.Board(rec["fen"].decode()).mirror()
        legal = {policy_to_idx[vocab_key(m)] for m in board.legal_moves}
        assert int(flipped.played_idx[i]) in legal


def test_dense_policy_target_sums_to_one():
    batch = records_to_batch(synth_records(n_games=3), mirror_prob=0.5)
    target = dense_policy_target(batch)
    np.testing.assert_allclose(target.sum(dim=1).numpy(), 1.0, rtol=1e-5)
    # The synthesized played move is always one of the visited moves.
    assert torch.all(target.gather(1, batch.played_idx[:, None]) > 0)


def test_window_growth():
    assert window_size(10, 0, 0.75, 0.4) == 0
    assert window_size(1000, 250_000, 0.75, 0.4) == 1000
    w1 = window_size(1_000_000, 250_000, 0.75, 0.4)
    w2 = window_size(10_000_000, 250_000, 0.75, 0.4)
    assert 250_000 < w1 < w2 < 10_000_000


def test_fifo_window_samples_newest(tmp_path):
    rows = synth_records(n_games=6)
    cap = len(rows)
    path = tmp_path / "buffer.bin"
    rows.tofile(path)
    total = cap + 37  # ring wrapped: newest record sits at index 36
    (tmp_path / "buffer.bin.writes").write_text(f"{total}\n")
    (tmp_path / "buffer.bin.stats.json").write_text(f'{{"capacity": {cap}, "fifo": true}}')
    buf = ReplayBuffer(path, window_c=10, window_alpha=1.0, window_beta=0.0, seed=0)
    assert buf.current_window() == 10
    idx = buf.sample_indices(4000)
    expected = {(total - 1 - r) % cap for r in range(10)}
    assert set(idx.tolist()) == expected
    b = buf.sample_batch(64)
    assert decode_planes(b).shape == (64, 8, 8, 111)


def test_empty_buffer(tmp_path):
    p = tmp_path / "buf.bin"
    p.write_bytes(b"")
    buf = ReplayBuffer(p)
    assert buf.size == 0
    with pytest.raises(RuntimeError):
        buf.sample_indices(4)
