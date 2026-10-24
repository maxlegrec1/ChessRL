"""Render the dashboard headlessly against a synthetic run; it must not raise."""

import json
import time

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_dashboard_renders(tmp_path, monkeypatch):
    t0 = time.time() - 3600
    run = tmp_path / "runs" / "20261021_200000_ab12_visits"
    _write_jsonl(run / "metrics" / "train.jsonl", [
        {"event": "start", "step": 0, "t": t0},
        *[{"step": s, "t": t0 + 10 * s, "loss/total": 3.0 - s / 100, "loss/policy_visits": 2.0,
           "policy/kl_to_visits": 0.5, "loss/value": 1.0, "loss/value_q": 0.9, "acc/policy_top_visit": 0.3,
           "entropy/policy": 2.0, "entropy/visit_target": 1.5, "lr": 1e-3, "buffer/decisive_outcome": 0.2,
           "buffer/wdl_agreement": 0.7, "buffer/castle_rate": 0.01, "buffer/frac_near_uniform": 0.3,
           "pacing/k_actual": 5.0, "pacing/step_time_s": 0.05, "buffer/window": 1000 + s, "params": 1285921,
           "model_blocks": 4, "model_channels": 64, "buffer/total_writes": 100 * s} for s in range(1, 50)],
        {"event": "grow", "step": 25, "t": t0 + 250, "spec": {"blocks": 6}},
    ])
    _write_jsonl(run / "metrics" / "selfplay.jsonl", [
        {"t": t0 + 30 * i, "positions_per_sec": 300.0, "nodes_per_sec": 2e5, "games_per_sec": 2.0,
         "total_games": 10 * i, "total_white_wins": 3 * i, "total_black_wins": 3 * i, "total_game_plies": 800 * i,
         "total_end_checkmate": 5 * i, "total_end_resign": i, "total_end_max_plies": i,
         "total_records_written": 1000 * i} for i in range(1, 20)
    ])
    _write_jsonl(run / "ladder" / "ladder.jsonl", [
        {"t": t0 + 600 * i, "step": 50 * i, "elo": 100.0 * i, "elo_se": 40.0,
         "puzzles": {"line": 0.05 * i, "first_move": 0.08 * i}, "anchors": {"random": {"w": 40, "d": 4, "l": 4}}}
        for i in range(1, 5)
    ])
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"t": time.time(), "spent": 12.3, "cap": 200, "launch_cap": 180,
                                  "burn_per_hour": 1.1, "hours_left": 170.6,
                                  "instances": [{"label": "dev1", "gpu": "RTX 4090", "dph": 0.6,
                                                 "started": t0, "ended": None}]}))
    monkeypatch.setenv("CHESSRL_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("CHESSRL_BUDGET", str(budget))
    at = AppTest.from_file("../scripts/dashboard.py", default_timeout=60).run()
    assert not at.exception, at.exception
    # Written just now, so it counts as running and is shown by default.
    assert any("visits" in str(df.value) for df in at.dataframe)
    by_label = {b.label: b for b in at.sidebar.button}
    by_label["None"].click().run()
    assert not at.exception, at.exception
    assert not any("visits" in str(df.value) for df in at.dataframe)
    {b.label: b for b in at.sidebar.button}["All"].click().run()
    assert not at.exception, at.exception
    assert any("visits" in str(df.value) for df in at.dataframe)
