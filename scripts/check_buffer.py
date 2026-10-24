"""Validate v2 records written by the C++ self-play against python-chess.

  uv run python scripts/check_buffer.py artifacts/runs/<id>/buffer/buffer.bin [-n 2000]

Checks per sampled record: version byte, current bitboards / castling / en
passant vs the FEN, decoded planes vs fen_encoder.encode_record_planes,
legality of every visited move and of the played move, visit fractions
summing to ~1, one-hot outcome. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.small.fen_encoder import encode_record_planes  # noqa: E402
from src.models.small.vocab import policy_index, policy_to_idx  # noqa: E402
from src.train.dataset import FLAG_FULL_SEARCH, RECORD_DTYPE, decode_planes, records_to_batch  # noqa: E402

_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def legal_indices(board: chess.Board) -> set[int]:
    out = set()
    for m in board.legal_moves:
        uci = m.uci()
        out.add(policy_to_idx.get(uci[:-1] if uci.endswith("n") else uci, -1))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("buffer", type=Path)
    p.add_argument("-n", type=int, default=2000)
    a = p.parse_args()
    mm = np.memmap(a.buffer, dtype=RECORD_DTYPE, mode="r")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(mm), size=min(a.n, len(mm)), replace=False))
    rows = np.asarray(mm[idx])
    fails: Counter = Counter()
    stats: Counter = Counter()

    batch = records_to_batch(rows, mirror_prob=0.0)
    planes = decode_planes(batch).numpy()
    for i, rec in enumerate(rows):
        if rec["version"] != 2:
            fails["version"] += 1
            continue
        fen = rec["fen"].decode()
        board = chess.Board(fen)
        bb = [int(board.pieces_mask(t, c)) for c in (chess.WHITE, chess.BLACK) for t in _TYPES]
        if [int(x) for x in rec["cur_bb"]] != bb:
            fails["cur_bb"] += 1
        cr = board.castling_rights
        castling = (bool(cr & chess.BB_H1) | bool(cr & chess.BB_A1) << 1 | bool(cr & chess.BB_H8) << 2
                    | bool(cr & chess.BB_A8) << 3)
        if int(rec["castling"]) != castling:
            fails["castling"] += 1
        ep_field = fen.split()[3]
        if int(rec["ep"]) != (255 if ep_field == "-" else chess.parse_square(ep_field)):
            fails["ep"] += 1
        ref = encode_record_planes(fen, past_bitboards=rec["past_bb"], past_repetitions=rec["past_rep"],
                                   current_repetition=bool(rec["cur_rep"]))
        if not np.array_equal(planes[i], ref):
            fails["planes"] += 1
        legal = legal_indices(board)
        k = int(rec["n_policy"])
        entries = rec["policy"][:k]
        if k == 0:
            fails["no_policy"] += 1
        if any(int(e[0]) not in legal for e in entries):
            fails["illegal_visit"] += 1
        total = entries[:, 1].astype(np.float64).sum() / 65535.0
        stats["visit_mass_lt_0.98"] += total < 0.98
        if total > 1.01:
            fails["visit_mass_gt_1"] += 1
        if int(rec["played_idx"]) not in legal:
            fails["illegal_played"] += 1
        if policy_index[int(rec["played_idx"])] not in (rec["played"].decode(), rec["played"].decode()[:-1]):
            fails["played_mismatch"] += 1
        vo = rec["value_outcome"]
        if sorted(vo.tolist()) != [0.0, 0.0, 1.0]:
            fails["outcome_not_onehot"] += 1
        stats["full_search"] += bool(rec["flags"] & FLAG_FULL_SEARCH)
        stats["entries"] += k
        stats["sims"] += int(rec["sims"])
        stats["ply"] += int(rec["ply"])
    n = len(rows)
    print(f"checked {n} records from {a.buffer} ({len(mm):,} in buffer)")
    print(f"  mean visited moves {stats['entries'] / n:.1f}, mean sims {stats['sims'] / n:.0f}, "
          f"mean ply {stats['ply'] / n:.1f}, full-search {stats['full_search'] / n:.0%}, "
          f"visit mass < 0.98 in {stats['visit_mass_lt_0.98'] / n:.1%} (top-20 truncation)")
    if fails:
        print("  FAILURES:", dict(fails))
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
