"""v3 replay buffer: canonical record layout, vectorized plane decode, window sampling.

Records are synthesized in Python from random python-chess games with the same
byte layout cpp/selfplay.cpp writes, canonical as it writes them: a position with
black to move is stored as white sees it (python-chess Board.mirror()). Decoded
planes must match the reference per-record encoder (fen_encoder.encode_record_planes)
and fen_to_tensor of the game's own position exactly.
"""

import random

import bulletchess
import chess
import numpy as np
import pytest
import torch

from src.models.small.fen_encoder import encode_record_planes, fen_to_tensor
from src.models.small.vocab import canonical_key, policy_to_idx
from src.train.dataset import (
    FLAG_BLACK_TO_MOVE,
    FLAG_FULL_SEARCH,
    RECORD_DTYPE,
    RECORD_SIZE,
    RECORD_VERSION,
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


def synth_records(n_games: int = 12, seed: int = 0, games: list | None = None) -> np.ndarray:
    """Canonical records of random games. `games`, when given, receives per record the game's own position
    (FEN, the 7 previous positions' FENs newest first, their repetition flags, the current one's)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_games):
        board = chess.Board()
        past: list[chess.Board] = []
        for ply in range(rng.randint(20, 90)):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            white = board.turn == chess.WHITE
            view = board if white else board.mirror()  # the record's view: white to move

            def key(m, white=white):
                return canonical_key(m.uci(), white)

            rec = np.zeros((), dtype=RECORD_DTYPE)
            fen = view.fen()
            rec["fen"] = fen.encode()
            move = rng.choice(legal)
            rec["played"] = key(move).encode()
            rec["value_outcome"] = [0, 1, 0]
            rec["value_mcts"] = [0.2, 0.5, 0.3]
            rec["plies_remaining"] = 10
            rec["backed_plies"] = 12
            past_reps = [rng.random() < 0.1 for _ in past[:7]]
            for k, b in enumerate(past[:7]):
                rec["past_bb"][k] = bitboards(b if white else b.mirror())
                rec["past_rep"][k] = past_reps[k]
            rec["cur_rep"] = rng.random() < 0.1
            rec["cur_bb"] = bitboards(view)
            rec["version"] = RECORD_VERSION
            rec["flags"] = FLAG_FULL_SEARCH | (0 if white else FLAG_BLACK_TO_MOVE)
            cr = view.castling_rights
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
                rec["policy"][k] = (policy_to_idx[key(m)], int(round(p * 65535)))
            rec["played_idx"] = policy_to_idx[key(move)]
            rows.append(rec)
            if games is not None:
                games.append((board.fen(), [b.fen() for b in past[:7]], past_reps, bool(rec["cur_rep"])))
            past.insert(0, board.copy(stack=False))
            board.push(move)
    return np.array(rows, dtype=RECORD_DTYPE)


def test_record_dtype_is_1280_bytes():
    assert RECORD_DTYPE.itemsize == RECORD_SIZE == 1280


def test_decode_matches_reference_encoder():
    rows = synth_records()
    planes = decode_planes(records_to_batch(rows)).numpy()
    for i, rec in enumerate(rows):
        ref = encode_record_planes(
            rec["fen"].decode(),
            past_bitboards=rec["past_bb"],
            past_repetitions=rec["past_rep"],
            current_repetition=bool(rec["cur_rep"]),
        )
        np.testing.assert_array_equal(planes[i], ref, err_msg=f"row {i} fen={rec['fen']}")


def test_records_are_what_the_encoder_makes_of_the_game():
    # Two independent canonical views: the record's (python-chess Board.mirror()) and fen_to_tensor's (planes
    # mirrored), from the game's own position and history.
    games: list = []
    rows = synth_records(n_games=6, seed=2, games=games)
    planes = decode_planes(records_to_batch(rows)).numpy()
    assert any(rec["flags"] & FLAG_BLACK_TO_MOVE for rec in rows)
    for i, (fen, past, past_reps, cur_rep) in enumerate(games):
        ref = fen_to_tensor(fen, [bulletchess.Board.from_fen(f) for f in past], past_reps, cur_rep)
        np.testing.assert_array_equal(planes[i], ref, err_msg=f"row {i} fen={fen}")


def test_targets_are_legal_in_the_record_position():
    for rec in synth_records(n_games=4, seed=3):
        board = chess.Board(rec["fen"].decode())
        assert board.turn == chess.WHITE
        legal = {policy_to_idx[canonical_key(m.uci(), True)] for m in board.legal_moves}
        assert int(rec["played_idx"]) in legal
        assert {int(e[0]) for e in rec["policy"][: int(rec["n_policy"])]} <= legal


def test_dense_policy_target_sums_to_one():
    batch = records_to_batch(synth_records(n_games=3))
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
