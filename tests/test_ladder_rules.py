"""Rating games end under the self-play rules: a threefold repetition is an automatic draw
when it happens, not when it merely could be claimed with the next move."""

import chess

from src.rating.ladder import rules_end


def test_threefold_ends_the_game_when_it_happens_not_when_claimable():
    b = chess.Board()
    for u in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        b.push_uci(u)
    assert b.can_claim_threefold_repetition()  # ...Ng8 would be the third occurrence
    assert rules_end(b) == (False, None)
    b.push_uci("f6g8")
    assert rules_end(b) == (True, None)


def test_checkmate_and_fifty_move_rule():
    b = chess.Board()
    for u in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate
        b.push_uci(u)
    assert rules_end(b) == (True, chess.BLACK)
    assert rules_end(chess.Board("8/8/8/4k3/8/8/8/4K2R w - - 100 80")) == (True, None)
    assert rules_end(chess.Board("8/8/8/4k3/8/8/8/4K2R w - - 99 80")) == (False, None)
