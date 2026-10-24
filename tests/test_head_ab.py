import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import head_ab  # noqa: E402

from src.rating import ladder  # noqa: E402


def test_head_to_head_never_searches_a_finished_game(monkeypatch):
    rng = random.Random(0)

    def fake_moves(engine, games, sims):
        out = []
        for g in games:
            assert not ladder.rules_end(g.board)[0], "searched a finished game"
            out.append(rng.choice(list(g.board.legal_moves)).uci())
        return out

    monkeypatch.setattr(ladder, "mcts_moves", fake_moves)
    openings = ladder.make_openings(40, seed=5)
    scores, histories = head_ab.head_to_head("a", "b", openings, sims=1, max_plies=400)
    assert len(scores) == len(histories) == 80
    assert all(s in (0.0, 0.5, 1.0) for s in scores)
    # Each game alternates engines: colours swapped per opening.
    assert len(head_ab.sample_positions(histories, 50)) == 50


def test_elo_diff_is_symmetric():
    assert abs(head_ab.elo_diff(0.5)) < 1e-9
    assert abs(head_ab.elo_diff(0.64) + head_ab.elo_diff(0.36)) < 1e-9
