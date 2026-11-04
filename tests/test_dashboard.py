"""The dashboard server (scripts/dashboard.py) on synthetic runs: incremental reads, payloads, branch offsets
and the HTTP endpoints."""

import importlib.util
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "dashboard", Path(__file__).resolve().parents[1] / "scripts" / "dashboard.py"
)
dashboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dashboard)


def _write_jsonl(path, rows, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))


def _make_run(root, name, t0, steps=range(1, 50), step0=0):
    run = root / name
    _write_jsonl(
        run / "metrics" / "train.jsonl",
        [
            {"event": "start", "step": step0, "t": t0},
            *[
                {
                    "step": step0 + s,
                    "t": t0 + 10 * s,
                    "loss/policy_visits": 2.0,
                    "policy/kl_to_visits": 0.5,
                    "loss/value": 1.0 - s / 100,
                    "loss/value_q": 0.9,
                    "lr": 1e-3,
                    "buffer/window": 1000 + s,
                    "params": 812556,
                    "model_d_model": 128,
                    "model_layers": 4,
                }
                for s in steps
            ],
            {"event": "grow", "step": step0 + 25, "t": t0 + 250},
        ],
    )
    _write_jsonl(
        run / "metrics" / "selfplay.jsonl",
        [
            {
                "t": t0 + 30 * i,
                "positions_per_sec": 300.0,
                "games_per_sec": 2.0,
                "nodes_per_sec": 2e5,
                "total_games": 10 * i,
                "total_white_wins": 3 * i,
                "total_black_wins": 3 * i,
                "total_draws": 4 * i,
                "total_game_plies": 800 * i,
                "total_end_checkmate": 5 * i,
                "total_end_resign": i,
                "total_end_max_plies": i,
                "total_records_written": 1000 * i,
            }
            for i in range(1, 20)
        ],
    )
    _write_jsonl(
        run / "ladder" / "ladder.jsonl",
        [
            {
                "t": t0 + 120 * i,
                "step": step0 + 10 * i,
                "elo": 100.0 * i,
                "elo_se": 40.0,
                "openings": "sf",
                "puzzles": {"line": 0.05 * i, "first_move": 0.08 * i},
                "opening": {
                    "policy": {"e2e4": 0.4, "d2d4": 0.3, "g1f3": 0.1, "a2a3": 0.1, "h2h3": 0.1},
                    "search": {"e2e4": 0.9, "b2b3": 0.1},
                },
            }
            for i in range(1, 5)
        ],
    )
    _write_jsonl(
        run / "metrics" / "diversity.jsonl",
        [
            {
                "t": t0 + 300,
                "plies": {
                    "8": {"records": 900, "effective": 850.0, "top_share": 0.01, "singletons": 0.9}
                },
            }
        ],
    )
    (run / "config.yaml").write_text(
        "search_heads: outcome\nselfplay:\n  random_opening_plies: 8\n"
    )
    return run


def test_table_reads_only_new_bytes_and_survives_a_replaced_file(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"t": 1, "a": 1}\n{"t": 2, "a": 2}\n{"t": 3, "a"')  # the last line unfinished
    table = dashboard.Table(path, ("a",))
    table.update()
    assert list(table.array("a")) == [1, 2]
    with path.open("a") as fh:
        fh.write(': 3}\n{"t": 4, "a": "x"}\n')
    table.update()
    assert list(table.times) == [1, 2, 3, 4]
    assert np.isnan(table.array("a")[-1])  # not a number
    # The home sync replaces the file whole (a new inode) with the same bytes plus more: continued.
    tmp = tmp_path / "m.new"
    tmp.write_bytes(path.read_bytes() + b'{"t": 5, "a": 5}\n')
    tmp.replace(path)
    table.update()
    assert list(table.times) == [1, 2, 3, 4, 5]
    # A different file under the same name: read again from the start.
    tmp.write_text('{"t": 9, "a": 9}\n' * 8)
    tmp.replace(path)
    table.update()
    assert list(table.times) == [9] * 8


def test_payload_series_kpis_and_derived_rates(tmp_path):
    t0 = time.time() - 3600
    run = _make_run(tmp_path / "runs", "20261021_200000_ab12_visits", t0)
    r = dashboard.Run("x", run)
    p = r.payload("hours", 5, 20)
    assert len(p["train"]["x"]) <= 20 and p["train"]["x"] == sorted(p["train"]["x"])
    assert p["train"]["cols"]["search_q_weight"][0] == 0.0  # search_heads outcome
    assert p["selfplay"]["cols"]["game_length"][-1] == 80.0  # 800 plies per 10 games
    assert p["selfplay"]["cols"]["frac_draw"][-1] == 0.4
    assert p["selfplay"]["cols"]["random_opening_plies"][-1] == 8.0  # from the config
    assert p["ladder"]["elo"] == [100.0, 200.0, 300.0, 400.0] and p["ladder"]["set"][0] == "sf"
    assert set(p["opening"]["policy"]["moves"]) == {"e2e4", "d2d4", "g1f3", "a2a3"}
    assert p["opening"]["policy"]["other"][0] == 0.1
    assert p["diversity"]["plies"]["8"]["effective"] == [850.0]
    assert len(p["events"]) == 1
    k = p["kpi"]
    assert k["step"] == 49 and k["model"] == "enc 128x4" and k["elo"] == 400.0 and k["games"] == 190
    for xkey in ("step", "positions"):
        assert r.payload(xkey, 1, 600)["train"]["x"]


def test_a_branch_continues_its_parent(tmp_path):
    root = tmp_path / "runs"
    t0 = time.time() - 7200
    parent = _make_run(root, "20261101_000000_aaaa_parent", t0)
    child = _make_run(root, "20261101_010000_0000_child", t0 + 3600, step0=49)
    (child / "metrics" / "branch.json").write_text(json.dumps({"parent_run": parent.name}))
    dash = dashboard.Dashboard([root], tmp_path / "none.json")
    runs = {r["label"]: r["id"] for r in dash.list_payload()["runs"]}
    out = dash.data_payload([runs["parent · aaaa"], runs["child · 0000"]], "hours", 1, 600)["runs"]
    parent_end = out[0]["kpi"]["hours"]
    assert out[1]["train"]["x"][0] >= parent_end - 0.01
    assert (
        out[1]["kpi"]["positions"] == 19000
    )  # its own counter; the positions axis adds the parent's
    pos = dash.data_payload([runs["child · 0000"]], "positions", 1, 600)["runs"][0]
    assert pos["train"]["x"][0] > 19000


def test_http_endpoints(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "20261021_200000_ab12_visits", time.time() - 600)
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            {"t": time.time(), "spent": 1.0, "cap": 300, "burn_per_hour": 0.5, "instances": []}
        )
    )
    dash = dashboard.Dashboard([root], budget)
    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(dash, ""))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "uPlot" in page and "/api/data" in page
        runs = json.load(urllib.request.urlopen(base + "/api/runs"))["runs"]
        assert runs[0]["running"] and runs[0]["elo"] == 400.0
        data = json.load(
            urllib.request.urlopen(f"{base}/api/data?ids={runs[0]['id']}&x=step&k=3&n=100")
        )
        assert data["budget"]["cap"] == 300 and data["runs"][0]["train"]["x"]
        assert json.load(urllib.request.urlopen(base + "/api/data?ids=nope"))["runs"] == []
    finally:
        server.shutdown()


def test_rolling_mean_and_reduce_points():
    y = np.array([1.0, np.nan, 3.0, 5.0])
    assert list(dashboard.rolling_mean(y, 2)) == [1.0, 1.0, 3.0, 4.0]
    x = np.arange(1000.0)
    xs, ys = dashboard.reduce_points(x, {"y": x * 2}, 100)
    assert len(xs) <= 100 and xs[-1] == 999.0 and ys["y"][-1] == 1998.0
    assert dashboard.rounded([123456.7, 0.000123456, float("nan")]) == [123500.0, 0.0001235, None]
