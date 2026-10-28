import chess
import pytest

from src.rating import ladder


def test_sf_openings_are_legal_and_distinct():
    ops = ladder.load_openings("sf", 50)
    assert len(ops) == 50 and len({tuple(o) for o in ops}) == 50
    for moves in ops:
        b = chess.Board()
        for u in moves:
            assert chess.Move.from_uci(u) in b.legal_moves
            b.push_uci(u)
        assert not b.is_game_over()


def test_random_openings_unchanged():
    assert ladder.load_openings("random", 5) == ladder.make_openings(5)


def test_too_many_sf_openings():
    with pytest.raises(ValueError):
        ladder.load_openings("sf", 10_000)
