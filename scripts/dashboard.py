"""ChessRL training dashboard: a small HTTP server and one page (scripts/dashboard_ui/index.html).

    .venv/bin/python scripts/dashboard.py --port 8500        # http://localhost:8500

Reads every run under $CHESSRL_RUNS_ROOT (colon-separated; default artifacts/runs), on a box
<root>/<run>/ and in the synced copy at home <root>/<box>/<run>/:
  metrics/train.jsonl      trainer losses, signal statistics, pacing, growth events
  metrics/selfplay.jsonl   self-play throughput and cumulative game counters
  metrics/diversity.jsonl  opening diversity of the newest records (scripts/opening_diversity.py)
  ladder/ladder.jsonl      ladder Elo, puzzles and the first-move probe (scripts/ladder_daemon.py)
  config.yaml              the run's resolved config
and the campaign budget ($CHESSRL_BUDGET, default /workspace/budget.json, pushed by chessrl-vast guard).

Files are read incrementally: each request parses only the bytes added since the last one (the home sync
replaces files whole, so a file whose old bytes are unchanged is continued, not re-read). The page gets each
curve smoothed on the server and reduced to at most `n` points per run, so a refresh costs milliseconds and
the browser draws a few thousand points per chart whatever the length of the runs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import threading
import time
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).resolve().parent / "dashboard_ui" / "index.html"
RUNNING_WITHIN_S = 15 * 60  # self-play logs every 30 s; the home sync runs every 5 min
XKEYS = ("hours", "step", "positions")

# Trainer columns the page draws, and the smoothing each gets: None = the page's smoothing slider.
TRAIN_SMOOTH = {
    "loss/policy_visits": None,
    "policy/kl_to_visits": None,
    "loss/value": None,
    "loss/value_q": None,
    "acc/policy_top_visit": None,
    "entropy/policy": None,
    "entropy/visit_target": None,
    "buffer/decisive_outcome": None,
    "buffer/wdl_agreement": None,
    "buffer/castle_rate": None,
    "buffer/frac_near_uniform": None,
    "pacing/k_actual": None,
    "pacing/step_time_s": None,
    "lr": 1,
    "buffer/window": 1,
    "search_q_weight": 1,
}
TRAIN_KPI = ("params", "model_d_model", "model_layers", "model_blocks", "model_channels")
# Self-play rows every 30 s: rates from the counters' differences, a little smoothing.
SELFPLAY_RAW = (
    "engine_step",
    "positions_per_sec",
    "games_per_sec",
    "nodes_per_sec",
    "random_opening_plies",
    "resign_threshold",
)
COUNTERS = (
    "games",
    "game_plies",
    "white_wins",
    "black_wins",
    "draws",
    "end_checkmate",
    "end_resign",
    "end_max_plies",
    "records_written",
    "playthrough_would_resign",
    "playthrough_false_resign",
)
SELFPLAY_SMOOTH = {
    "positions_per_sec": 3,
    "games_per_sec": 3,
    "nodes_per_sec": 3,
    "random_opening_plies": 1,
    "resign_threshold": 1,
    "game_length": 5,
    "frac_checkmate": 5,
    "frac_resign": 5,
    "frac_max_plies": 5,
    "frac_draw": 5,
    "frac_false_resign": 10,
}
# The first-move probe: the four main moves of each run, the rest as "other".
OPENING_TOP = 4


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else math.nan


class Table:
    """A JSONL file read incrementally. With `cols`, rows become float columns (NaN where a row lacks
    the key); rows carrying an "event" key are kept apart. Without, whole rows are kept (small files)."""

    def __init__(self, path: Path, cols: tuple[str, ...] | None = None) -> None:
        self.path, self.cols = path, cols
        self._reset()

    def _reset(self) -> None:
        self.ino: int | None = None
        self.offset = 0
        self.tail = b""  # the last bytes read, to recognise the same file after a sync replaced it
        self.partial = b""  # an unfinished last line
        self.t = array("d")
        self.data = {c: array("d") for c in self.cols or ()}
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self.version = 0

    def update(self) -> None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            if self.offset:
                version = self.version
                self._reset()
                self.version = version + 1
            return
        if st.st_ino != self.ino:
            same = False
            if self.ino is not None and st.st_size >= self.offset and self.tail:
                with self.path.open("rb") as fh:
                    fh.seek(self.offset - len(self.tail))
                    same = fh.read(len(self.tail)) == self.tail
            if not same:
                version = self.version
                self._reset()
                self.version = version + 1
            self.ino = st.st_ino
        if st.st_size < self.offset:
            version = self.version
            self._reset()
            self.version, self.ino = version + 1, st.st_ino
        if st.st_size == self.offset:
            return
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read(st.st_size - self.offset)
        self.offset += len(chunk)
        self.tail = (self.tail + chunk)[-64:]
        lines = (self.partial + chunk).split(b"\n")
        self.partial = lines.pop()
        for line in lines:
            if line.strip():
                try:
                    self._add(json.loads(line))
                except ValueError:
                    pass
        self.version += 1

    def _add(self, row) -> None:
        if not isinstance(row, dict):
            return
        if self.cols is None:
            self.rows.append(row)
            return
        if "event" in row:
            self.events.append({k: row.get(k) for k in ("event", "t", "step")})
            return
        t = _num(row.get("t"))
        if t != t:
            return
        self.t.append(t)
        for c, col in self.data.items():
            col.append(_num(row.get(c)))

    # Copies: a view would keep the array exporting its buffer, and the next append would fail.
    def array(self, col: str) -> np.ndarray:
        return np.array(self.data[col], dtype=np.float64) if col in self.data else np.array([])

    @property
    def times(self) -> np.ndarray:
        return np.array(self.t, dtype=np.float64)


def rolling_mean(y: np.ndarray, k: int) -> np.ndarray:
    """The mean of the last k rows' values that are not NaN (pandas rolling(k, min_periods=1))."""
    if k <= 1 or len(y) == 0:
        return y
    ok = np.isfinite(y)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, y, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(1, len(y) + 1)
    j = np.maximum(0, i - k)
    n = cn[i] - cn[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[i] - cs[j]) / n, np.nan)


def reduce_points(x: np.ndarray, ys: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, dict]:
    """Bucket means of consecutive rows: at most n points, the last row kept as the last point."""
    if len(x) <= n:
        return x, ys
    edges = np.unique(np.linspace(0, len(x) - 1, n).astype(np.int64))
    starts = edges[:-1]

    def bucket(v):
        ok = np.isfinite(v)
        s = np.add.reduceat(np.where(ok, v, 0.0), starts)
        c = np.add.reduceat(ok.astype(np.int64), starts)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(c > 0, s / c, np.nan)
        return np.append(out, v[-1])

    return bucket(x), {k: bucket(v) for k, v in ys.items()}


def rounded(a) -> list:
    """Four significant digits, NaN as null: a fifth of the JSON."""
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a) & (a != 0)
    e = 3 - np.floor(np.log10(np.abs(np.where(ok, a, 1.0))))
    p = 10.0 ** np.minimum(np.abs(e), 22)  # exact powers of ten, so the results print short
    with np.errstate(invalid="ignore"):
        r = np.where(e >= 0, np.round(a * p) / p, np.round(a / p) * p)
    r = np.where(ok, r, a)
    return [v if v == v and v not in (math.inf, -math.inf) else None for v in r.tolist()]


def run_label(path: Path) -> str:
    # Run dirs are <date>_<time>_<hex>[_<name>]: "name (box)" when synced home under <root>/<box>/,
    # else "name · hex".
    parts = path.name.split("_")
    name = "_".join(parts[3:]) if len(parts) > 3 else path.name
    box = path.parent.name
    if box not in ("runs", ""):
        return f"{name} ({box})"
    return f"{name} · {parts[2]}" if len(parts) > 3 else name


def tail_json(path: Path) -> dict | None:
    """The last complete row of a JSONL file, read from its end."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 16384))
            lines = fh.read().split(b"\n")
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def q_weight_at(cfg: dict, steps: np.ndarray) -> np.ndarray:
    """The weight of the search-trained value head in the WDL the search reads (src/export/heads.py)."""
    heads = cfg.get("search_heads", "search")
    if heads == "outcome":
        return np.zeros_like(steps)
    if heads != "blend":
        return np.ones_like(steps)
    ramp = cfg.get("search_q_ramp") or {}
    if int(ramp.get("end_step", 0) or 0) <= 0:
        return np.full_like(steps, float(cfg.get("search_q_weight", 0.5)))
    a, b = float(ramp.get("start_step", 0)), float(ramp["end_step"])
    frac = np.clip((steps - a) / max(1.0, b - a), 0.0, 1.0)
    return (
        float(ramp.get("start", 0.0))
        + (float(ramp.get("end", 0.0)) - float(ramp.get("start", 0.0))) * frac
    )


class Run:
    """One run directory. A branch (another run's checkpoint continued here) names its parent in
    metrics/branch.json, {"parent_run": "<sibling run dir>"}: its hours and positions then continue from
    where the parent's ended (its steps already do). The older form {"parent_t0", "parent_positions"} gives
    the offsets directly."""

    def __init__(self, run_id: str, path: Path, resolve=None) -> None:
        self.id, self.path, self.label = run_id, path, run_label(path)
        self.resolve = resolve or (lambda path: None)  # a sibling run directory -> its Run
        m = path / "metrics"
        self.train = Table(m / "train.jsonl", ("step", *TRAIN_SMOOTH, *TRAIN_KPI))
        self.selfplay = Table(
            m / "selfplay.jsonl", (*SELFPLAY_RAW, *(f"total_{c}" for c in COUNTERS))
        )
        self.ladder = Table(path / "ladder" / "ladder.jsonl")
        self.diversity = Table(m / "diversity.jsonl")
        self.config: dict = {}
        self._config_mtime = 0.0
        self._cache: dict = {}

    def update(self) -> tuple:
        for t in (self.train, self.selfplay, self.ladder, self.diversity):
            t.update()
        try:
            mtime = (self.path / "config.yaml").stat().st_mtime
            if mtime != self._config_mtime:
                self.config = yaml.safe_load((self.path / "config.yaml").read_text()) or {}
                self._config_mtime = mtime
        except (FileNotFoundError, yaml.YAMLError):
            pass
        return (
            self.train.version,
            self.selfplay.version,
            self.ladder.version,
            self.diversity.version,
            self._config_mtime,
        )

    # ------------------------------------------------------------------ axes
    def _branch(self) -> dict:
        try:
            return json.loads((self.path / "metrics" / "branch.json").read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def offsets(self, depth: int = 0) -> tuple[float, float]:
        """(hours, positions) the parent run had reached when this branch started: 0 without a parent."""
        parent_name = self._branch().get("parent_run")
        parent = self.resolve(self.path.parent / parent_name) if parent_name and depth < 8 else None
        if parent is None:
            return 0.0, 0.0
        parent.update()
        axes, _ = parent._axes(depth + 1)
        t = parent.selfplay.times
        if not len(t):
            return 0.0, 0.0
        end = np.array([t[-1]])
        return float(axes["hours"](end)[0]), float(np.nan_to_num(axes["positions"](end)[0]))

    def _axes(self, depth: int = 0):
        """Functions t -> hours (training time: downtime > 10 min removed), positions, step."""
        branch = self._branch()
        off_h, off_p = self.offsets(depth)
        sp_t = self.selfplay.times
        tr_t = self.train.times
        starts = [a[0] for a in (tr_t, sp_t) if len(a)]
        starts += [e["t"] for e in self.train.events[:1] if isinstance(e.get("t"), (int, float))]
        own_t0 = min(starts) if starts else self.path.stat().st_mtime
        t0 = float(branch.get("parent_t0", own_t0)) - off_h * 3600.0
        ts = np.sort(sp_t)
        gaps = np.diff(ts)
        big = np.append(
            gaps > 600, True
        )  # the open interval after the last row counts once > 600 s
        down_s = np.concatenate([[0.0], np.cumsum(np.where(gaps > 600, gaps - 30, 0.0))])

        def hours(t):
            t = np.asarray(t, dtype=np.float64)
            down = np.zeros_like(t)
            if len(ts):
                i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, None)
                since = t - ts[i]
                inside = big[i] & (since > np.where(i == len(ts) - 1, 600, 0))
                down = down_s[i] + np.where(inside, np.maximum(0.0, since - 30), 0.0)
            return (t - t0 - down) / 3600.0

        recs = self.selfplay.array("total_records_written")
        if len(recs):
            ok = np.isfinite(recs)
            xs, ys = sp_t[ok], recs[ok]
            resets = np.where(ys[1:] < ys[:-1], ys[:-1], 0.0)  # the counter restarts with self-play
            ys = (
                ys
                + np.concatenate([[0.0], np.cumsum(resets)])
                + float(branch.get("parent_positions", off_p))
            )
        else:
            xs = ys = np.array([])

        def positions(t):
            return np.interp(t, xs, ys) if len(xs) else np.full(len(np.atleast_1d(t)), np.nan)

        steps = self.train.array("step")
        ok = np.isfinite(steps)
        st_t, st_v = tr_t[ok], steps[ok]

        def step(t):
            return np.interp(t, st_t, st_v) if len(st_t) else np.full(len(np.atleast_1d(t)), np.nan)

        return {"hours": hours, "positions": positions, "step": step}, own_t0

    # ------------------------------------------------------------------ payload
    def payload(self, xkey: str, k: int, n: int) -> dict:
        key = (self.update(), self.offsets(), xkey, k, n)
        if key in self._cache:
            return self._cache[key]
        axes, own_t0 = self._axes()
        out = {"id": self.id, "label": self.label, "config": self.config}
        out["train"] = self._train(axes, xkey, k, n)
        out["selfplay"] = self._selfplay(axes, xkey, n)
        lad = [
            r
            for r in self.ladder.rows
            if isinstance(r.get("t"), (int, float))
            and isinstance(r.get("elo"), (int, float))
            and r["t"] >= own_t0 - 1800  # a branch's copied rows are its parent's curve
        ]
        out["ladder"] = self._ladder(lad, axes, xkey)
        out["opening"] = self._opening(lad, axes, xkey)
        out["diversity"] = self._diversity(axes, xkey)
        out["events"] = [
            rounded(axes[xkey](np.array([e["t"]])) if xkey != "step" else [e.get("step") or 0])[0]
            for e in self.train.events
            if e.get("event") == "grow" and isinstance(e.get("t"), (int, float))
        ]
        out["kpi"] = self._kpi(lad, axes)
        self._cache = {key: out}
        return out

    def _train(self, axes, xkey, k, n) -> dict:
        tr = self.train
        t = tr.times
        if not len(t):
            return {"x": [], "cols": {}}
        steps = tr.array("step")
        x = steps if xkey == "step" else axes[xkey](t)
        cols = {}
        for c, fixed in TRAIN_SMOOTH.items():
            v = q_weight_at(self.config, steps) if c == "search_q_weight" else tr.array(c)
            if np.isfinite(v).any():
                cols[c] = rolling_mean(v, fixed or k)
        order = np.argsort(x, kind="stable")
        keep = np.isfinite(x[order])
        order = order[keep]
        x, cols = reduce_points(x[order], {c: v[order] for c, v in cols.items()}, n)
        return {"x": rounded(x), "cols": {c: rounded(v) for c, v in cols.items()}}

    def _selfplay(self, axes, xkey, n) -> dict:
        sp = self.selfplay
        t = sp.times
        if len(t) < 2:
            return {"x": [], "cols": {}}
        d = {c: np.diff(sp.array(f"total_{c}")) for c in COUNTERS}
        # A counter going down is a self-play restart; a counter older runs lack stays NaN.
        ok = np.all([np.nan_to_num(v, nan=0.0) >= 0 for v in d.values()], axis=0) & (d["games"] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(ok, d["games"], np.nan)
            would = np.where(
                ok & (d["playthrough_would_resign"] > 0), d["playthrough_would_resign"], np.nan
            )
            derived = {
                "game_length": d["game_plies"] / g,
                "frac_checkmate": d["end_checkmate"] / g,
                "frac_resign": d["end_resign"] / g,
                "frac_max_plies": d["end_max_plies"] / g,
                "frac_draw": d["draws"] / g,
                "frac_false_resign": d["playthrough_false_resign"] / would,
            }
        cols = {c: sp.array(c)[1:] for c in SELFPLAY_RAW}
        cols.update(derived)
        if not np.isfinite(cols["random_opening_plies"]).any():  # runs from before the column
            plies = (self.config.get("selfplay") or {}).get("random_opening_plies")
            if isinstance(plies, (int, float)):
                cols["random_opening_plies"] = np.full(len(t) - 1, float(plies))
        x = axes[xkey](t[1:])
        cols = {
            c: rolling_mean(v, SELFPLAY_SMOOTH.get(c, 1))
            for c, v in cols.items()
            if np.isfinite(v).any()
        }
        keep = np.isfinite(x)
        x, cols = reduce_points(x[keep], {c: v[keep] for c, v in cols.items()}, n)
        return {"x": rounded(x), "cols": {c: rounded(v) for c, v in cols.items()}}

    @staticmethod
    def _xs(rows, axes, xkey) -> np.ndarray:
        if xkey == "step":
            return np.array([_num(r.get("step")) for r in rows])
        return axes[xkey](np.array([r["t"] for r in rows], dtype=np.float64))

    def _ladder(self, lad, axes, xkey) -> dict:
        if not lad:
            return {
                "x": [],
                "elo": [],
                "elo_se": [],
                "set": [],
                "puzzles.line": [],
                "puzzles.first_move": [],
            }
        x = self._xs(lad, axes, xkey)
        order = np.argsort(x, kind="stable")
        rows = [lad[i] for i in order]

        def col(f):
            return rounded([f(r) for r in rows])

        return {
            "x": rounded(x[order]),
            "elo": col(lambda r: _num(r.get("elo"))),
            "elo_se": col(lambda r: _num(r.get("elo_se"))),
            # Rows without the field predate the Stockfish openings: random openings.
            "set": [r.get("openings") or "random" for r in rows],
            "puzzles.line": col(lambda r: _num((r.get("puzzles") or {}).get("line"))),
            "puzzles.first_move": col(lambda r: _num((r.get("puzzles") or {}).get("first_move"))),
        }

    def _opening(self, lad, axes, xkey) -> dict:
        out = {}
        rows = [r for r in lad if isinstance(r.get("opening"), dict)]
        if not rows:
            return out
        x = self._xs(rows, axes, xkey)
        order = np.argsort(x, kind="stable")
        rows = [rows[i] for i in order]
        for kind in ("policy", "search"):
            shares = [(r["opening"].get(kind) or {}) for r in rows]
            best: dict[str, float] = {}
            for s in shares:
                for move, p in s.items():
                    best[move] = max(best.get(move, 0.0), _num(p))
            top = sorted(best, key=best.get, reverse=True)[:OPENING_TOP]
            if not top:
                continue
            moves = {m: rounded([_num(s.get(m, 0.0)) for s in shares]) for m in top}
            other = [max(0.0, 1.0 - sum(_num(s.get(m, 0.0)) for m in top)) for s in shares]
            out[kind] = {"x": rounded(x[order]), "moves": moves, "other": rounded(other)}
        return out

    def _diversity(self, axes, xkey) -> dict:
        rows = [
            r
            for r in self.diversity.rows
            if isinstance(r.get("t"), (int, float)) and r.get("plies")
        ]
        if not rows:
            return {}
        x = (
            self._xs(rows, axes, xkey)
            if xkey != "step"
            else axes["step"](np.array([r["t"] for r in rows], dtype=np.float64))
        )
        plies = sorted({p for r in rows for p in r["plies"]}, key=int)
        out = {"x": rounded(x), "plies": {}}
        for p in plies:
            out["plies"][p] = {
                f: rounded([_num((r["plies"].get(p) or {}).get(f)) for r in rows])
                for f in ("effective", "top_share", "singletons")
            }
        return out

    def _kpi(self, lad, axes) -> dict:
        tr, sp = self.train, self.selfplay

        def last(table, col):
            a = table.array(col)
            a = a[np.isfinite(a)]
            return float(a[-1]) if len(a) else None

        games = last(sp, "total_games") or 0
        plies = last(sp, "total_game_plies") or 0
        hours_now = float(axes["hours"](np.array([sp.times[-1]]))[0]) if len(sp.times) else None
        d_model, layers = last(tr, "model_d_model"), last(tr, "model_layers")
        blocks, channels = last(tr, "model_blocks"), last(tr, "model_channels")
        if d_model:
            model = f"enc {int(d_model)}x{int(layers or 0)}"
        elif blocks:
            model = f"res {int(blocks)}x{int(channels or 0)}"
        else:
            model = None
        elos = [
            r for r in lad if r.get("openings", "random") == (lad[-1].get("openings") or "random")
        ]
        last_hour = []
        if lad and hours_now is not None:
            h = self._xs(elos, axes, "hours")
            last_hour = [r["elo"] for r, hh in zip(elos, h, strict=True) if hh >= hours_now - 1.0]
        return {
            "hours": hours_now,
            "step": last(tr, "step"),
            "model": model,
            "params": last(tr, "params"),
            "pos_s": last(sp, "positions_per_sec"),
            "games": games,
            "positions": last(sp, "total_records_written"),
            "avg_plies": plies / games if games else None,
            "elo": lad[-1]["elo"] if lad else None,
            "elo_se": _num(lad[-1].get("elo_se")) if lad else None,
            "elo_last_hour": sum(last_hour) / len(last_hour) if last_hour else None,
            "elo_last_hour_n": len(last_hour),
            "puzzles": _num((lad[-1].get("puzzles") or {}).get("line")) if lad else None,
        }


class Dashboard:
    def __init__(self, roots: list[Path], budget: Path) -> None:
        self.roots, self.budget = roots, budget
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()
        self._listed = (0.0, [])

    def discover(self) -> list[dict]:
        """Every run directory with metrics/: <root>/<run> on a box, <root>/<box>/<run> at home."""
        now = time.time()
        if now - self._listed[0] < 15:
            return self._listed[1]
        found = []
        for i, root in enumerate(self.roots):
            for metrics in sorted([*root.glob("*/metrics"), *root.glob("*/*/metrics")]):
                path = metrics.parent
                rid = (
                    str(path.relative_to(root))
                    if len(self.roots) == 1
                    else f"{i}/{path.relative_to(root)}"
                )
                sp = metrics / "selfplay.jsonl"
                activity = sp.stat().st_mtime if sp.exists() else metrics.stat().st_mtime
                lad = tail_json(path / "ladder" / "ladder.jsonl") or {}
                found.append(
                    {
                        "id": rid,
                        "path": path,
                        "label": run_label(path),
                        "activity": activity,
                        "running": now - activity < RUNNING_WITHIN_S,
                        "elo": lad.get("elo") if isinstance(lad.get("elo"), (int, float)) else None,
                    }
                )
        found.sort(key=lambda r: r["activity"], reverse=True)
        self._listed = (now, found)
        return found

    def list_payload(self) -> dict:
        return {"runs": [{k: v for k, v in r.items() if k != "path"} for r in self.discover()]}

    def data_payload(self, ids: list[str], xkey: str, k: int, n: int) -> dict:
        known = {r["id"]: r["path"] for r in self.discover()}
        by_path = {p: rid for rid, p in known.items()}

        def run(rid: str) -> Run:
            if rid not in self.runs:
                self.runs[rid] = Run(
                    rid, known[rid], lambda path: run(by_path[path]) if path in by_path else None
                )
            return self.runs[rid]

        out = []
        with self.lock:
            for rid in ids:
                if rid in known:
                    out.append(run(rid).payload(xkey, k, n))
        return {"runs": out, "budget": self.budget_payload(), "t": time.time()}

    def budget_payload(self) -> dict | None:
        try:
            return json.loads(self.budget.read_text())
        except (FileNotFoundError, ValueError):
            return None


def make_handler(dash: Dashboard, analysis_url: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet: the page polls every minute
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            gz = "gzip" in (self.headers.get("Accept-Encoding") or "") and len(body) > 1024
            if gz:
                body = gzip.compress(body, compresslevel=5)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(
                json.dumps(obj, separators=(",", ":"), default=str).encode(), "application/json"
            )

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/runs":
                    self._json({**dash.list_payload(), "analysis_url": analysis_url})
                elif url.path == "/api/data":
                    ids = [i for i in (q.get("ids", [""])[0]).split(",") if i]
                    xkey = q.get("x", ["hours"])[0]
                    xkey = xkey if xkey in XKEYS else "hours"
                    k = max(1, min(500, int(q.get("k", ["20"])[0])))
                    n = max(50, min(5000, int(q.get("n", ["600"])[0])))
                    self._json(dash.data_payload(ids, xkey, k, n))
                else:
                    self.send_error(404)
            except BrokenPipeError:
                pass
            except Exception as exc:  # report it to the page instead of dropping the connection
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--analysis-url",
        default=os.environ.get("CHESSRL_ANALYSIS_URL", ""),
        help="link to the analysis board in the header (empty: none)",
    )
    a = p.parse_args()
    roots = [
        Path(x)
        for x in os.environ.get("CHESSRL_RUNS_ROOT", str(ROOT / "artifacts" / "runs")).split(":")
    ]
    budget = Path(os.environ.get("CHESSRL_BUDGET", "/workspace/budget.json"))
    dash = Dashboard(roots, budget)
    server = ThreadingHTTPServer((a.host, a.port), make_handler(dash, a.analysis_url))
    print(f"[dashboard] http://localhost:{a.port}  runs: {', '.join(map(str, roots))}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
