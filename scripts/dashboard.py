"""ChessRL training dashboard (read-only).

Reads every run under artifacts/runs/ (or $CHESSRL_RUNS_ROOT):
  metrics/train.jsonl     trainer losses, signal stats, pacing, growth events
  metrics/selfplay.jsonl  self-play throughput + cumulative game statistics
  ladder/ladder.jsonl     anchor-ladder Elo + puzzle accuracy (scripts/ladder_daemon.py)
and the campaign budget from /workspace/budget.json (pushed by chessrl-vast guard).

  uv run streamlit run scripts/dashboard.py --server.port 8501 --server.headless true
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
# One or more run roots, colon-separated (e.g. the next-round and main checkouts).
RUNS_ROOTS = [
    Path(p)
    for p in os.environ.get("CHESSRL_RUNS_ROOT", str(ROOT / "artifacts" / "runs")).split(":")
]
BUDGET = Path(os.environ.get("CHESSRL_BUDGET", "/workspace/budget.json"))
PALETTE = ["#3b82f6", "#f97316", "#10b981", "#e11d48", "#8b5cf6", "#eab308", "#06b6d4", "#64748b"]

st.set_page_config(page_title="ChessRL dashboard", page_icon="♟", layout="wide")


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # partially written last line
    return pd.json_normalize(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=55)
def load_run(run_dir: str) -> dict:
    d = Path(run_dir)
    train = read_jsonl(d / "metrics" / "train.jsonl")
    events = train[train["event"].notna()] if "event" in train else pd.DataFrame()
    if "event" in train:
        train = train[train["event"].isna()]
    selfplay = read_jsonl(d / "metrics" / "selfplay.jsonl")
    lad = read_jsonl(d / "ladder" / "ladder.jsonl")
    # A branch (another run's checkpoint + ladder history, continued here) carries
    # metrics/branch.json {"parent_t0", "parent_positions"}: its curves continue the
    # parent's hours / positions axes. The copied ladder rows are the parent's own
    # curve: drop them here (they plotted at negative hours).
    try:
        branch = json.loads((d / "metrics" / "branch.json").read_text())
    except (FileNotFoundError, ValueError):
        branch = {}
    starts = [df["t"].min() for df in (train, selfplay, events) if not df.empty and "t" in df]
    own_t0 = min(starts) if starts else d.stat().st_mtime
    if not lad.empty and "t" in lad:
        lad = lad[
            lad["t"] >= own_t0 - 1800
        ].copy()  # a run's first rating may precede its metrics a little
    t0 = float(branch.get("parent_t0", own_t0))
    # Hours of training, not of wall clock: a run stopped and resumed later (or down
    # after a crash) would otherwise jump right by its downtime. Self-play logs every
    # 30 s; any longer silence counts as downtime.
    ts = (
        np.sort(selfplay["t"].to_numpy(dtype=float))
        if not selfplay.empty and "t" in selfplay
        else np.array([])
    )
    gaps = np.diff(ts)
    big = np.append(
        gaps > 600, True
    )  # the open interval after the last row counts once it is > 600 s
    down_s = np.concatenate([[0.0], np.cumsum(np.where(gaps > 600, gaps - 30, 0.0))])
    for df in (train, selfplay, events, lad):
        if not df.empty and "t" in df:
            t = df["t"].to_numpy(dtype=float)
            down = np.zeros_like(t)
            if len(ts):
                i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, None)
                since = t - ts[i]
                # Inside a gap (or past the last row by > 600 s) the extra time is downtime.
                inside = big[i] & (since > np.where(i == len(ts) - 1, 600, 0))
                down = down_s[i] + np.where(inside, np.maximum(0.0, since - 30), 0.0)
            df["hours"] = (t - t0 - down) / 3600.0
    # Self-play positions written so far, for every table (interpolated on time).
    if not selfplay.empty and "total_records_written" in selfplay:
        xs = selfplay["t"].to_numpy(dtype=float)
        ys = selfplay["total_records_written"].to_numpy(dtype=float)
        # The counter restarts at 0 with each self-play process: keep it cumulative.
        resets = np.where(ys[1:] < ys[:-1], ys[:-1], 0.0)
        ys = (
            ys
            + np.concatenate([[0.0], np.cumsum(resets)])
            + float(branch.get("parent_positions", 0))
        )
        for df in (train, events, lad):
            if not df.empty and "t" in df:
                df["positions"] = np.interp(df["t"].to_numpy(dtype=float), xs, ys)
        selfplay["positions"] = ys
    cfg = {}
    if (d / "config.yaml").exists():
        cfg = yaml.safe_load((d / "config.yaml").read_text()) or {}
    return {
        "train": train,
        "selfplay": selfplay,
        "events": events,
        "ladder": lad,
        "config": cfg,
        "t0": t0,
        "name": d.name,
    }


@st.cache_data(ttl=55)
def gather(name: str) -> pd.DataFrame:
    """Rows of every run's metrics/<name>: <root>/<run>/metrics on a box,
    <root>/<box>/<run>/metrics in the synced copy at home."""
    files = [
        p
        for root in RUNS_ROOTS
        for pat in (f"*/metrics/{name}", f"*/*/metrics/{name}")
        for p in root.glob(pat)
    ]
    return pd.concat([read_jsonl(p) for p in files] or [pd.DataFrame()], ignore_index=True)


def rerated_points(data: dict, rerate: pd.DataFrame, xkey: str) -> pd.DataFrame:
    """A run's checkpoints re-rated on the Stockfish openings (scripts/rerate.py, which may have
    run on another box), placed on the run's x axis through its trainer rows (step -> hours, positions)."""
    tr = data["train"]
    if rerate.empty or "run" not in rerate or tr.empty or "step" not in tr or xkey not in tr:
        return pd.DataFrame()
    rr = (
        rerate[rerate["run"] == data["name"]]
        .drop_duplicates("step", keep="last")
        .sort_values("step")
    )
    tr = tr.sort_values("step").drop_duplicates("step", keep="last")  # a resume repeats a few steps
    steps = tr["step"].to_numpy(dtype=float)
    # A branch's own rows start at its branch point; the last checkpoint may follow the last log row.
    rr = rr[(rr["step"] >= steps.min() - 100) & (rr["step"] <= steps.max() + 1000)].copy()
    if xkey != "step":
        rr[xkey] = np.interp(
            rr["step"].to_numpy(dtype=float), steps, tr[xkey].to_numpy(dtype=float)
        )
    return rr


def list_runs() -> list[Path]:
    runs = [p for root in RUNS_ROOTS if root.exists() for p in root.rglob("metrics") if p.is_dir()]
    return sorted({p.parent for p in runs}, key=lambda p: p.name)  # names start with the start time


def smooth(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=1).mean() if k > 1 else s


def line_chart(
    title: str,
    runs: dict,
    table: str,
    cols: list[str],
    xkey: str,
    k: int,
    yfmt: str | None = None,
    height: int = 300,
    log_y: bool = False,
) -> None:
    fig = go.Figure()
    dashes = ["solid", "dot", "dash", "longdash"]
    for i, (name, data) in enumerate(runs.items()):
        df = data[table]
        if df.empty or xkey not in df:
            continue
        for j, col in enumerate(cols):
            if col not in df:
                continue
            y = smooth(pd.to_numeric(df[col], errors="coerce"), k)
            label = name if len(cols) == 1 else f"{name} · {col.split('/')[-1]}"
            fig.add_trace(
                go.Scatter(
                    x=df[xkey],
                    y=y,
                    mode="lines",
                    name=label,
                    line=dict(color=PALETTE[i % len(PALETTE)], dash=dashes[j % 4], width=2),
                )
            )
        ev = data["events"]
        if not ev.empty and xkey in ev:
            for _, e in ev[ev["event"] == "grow"].iterrows():
                fig.add_vline(
                    x=e[xkey], line=dict(color=PALETTE[i % len(PALETTE)], dash="dot", width=1)
                )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.25),
        xaxis_title=XLABEL[xkey],
    )
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    if log_y:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")


XLABEL = {"hours": "hours of training", "step": "trainer step", "positions": "self-play positions"}


def run_label(path: Path) -> str:
    # Run dirs are <date>_<time>_<hex>[_<name>]; show "name (box)" when synced
    # home under ~/chessrl-runs/<box>/, else "name · hex".
    parts = path.name.split("_")
    name = "_".join(parts[3:]) if len(parts) > 3 else path.name
    box = path.parent.name
    if box not in ("runs", ""):
        return f"{name} ({box})"
    return f"{name} · {parts[2]}" if len(parts) > 3 else name


# Runs shown by default: everything still running, plus the run names in $CHESSRL_PINNED
# (comma-separated, e.g. "base-wd,my-run").
PINNED = {n.strip() for n in os.environ.get("CHESSRL_PINNED", "").split(",") if n.strip()}
RUNNING_WITHIN_S = 15 * 60  # self-play logs every 30 s; the home sync runs every 5 min


def last_activity(run_dir: Path) -> float:
    try:
        return (run_dir / "metrics" / "selfplay.jsonl").stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(ttl=55)
def last_elo(run_dir: str) -> float | None:
    try:
        lines = (Path(run_dir) / "ladder" / "ladder.jsonl").read_text().strip().splitlines()
        return json.loads(lines[-1])["elo"] if lines else None
    except (FileNotFoundError, ValueError, KeyError):
        return None


def select_runs(labels: dict[str, Path]) -> list[str]:
    """Sidebar run toggles. Defaults: running runs + PINNED; the choice survives refreshes."""
    now = time.time()
    running = {n for n, p in labels.items() if now - last_activity(p) < RUNNING_WITHIN_S}
    default = running | {n for n in labels if n.split(" ")[0] in PINNED}
    if not default:  # nothing running: the three most recent runs
        default = set(sorted(labels, key=lambda n: last_activity(labels[n]))[-3:])
    for n in labels:
        st.session_state.setdefault(f"show::{n}", n in default)

    def set_all(names, value):
        for n in labels:
            st.session_state[f"show::{n}"] = value if names is None else n in names

    c = st.sidebar.columns(3)
    c[0].button("Running", on_click=set_all, args=(running, True), width="stretch")
    c[1].button("All", on_click=set_all, args=(None, True), width="stretch")
    c[2].button("None", on_click=set_all, args=(None, False), width="stretch")

    def toggle(n):
        elo = last_elo(str(labels[n]))
        st.checkbox(f"{n} · {elo:,.0f}" if elo is not None else n, key=f"show::{n}")

    newest_first = sorted(labels, key=lambda n: last_activity(labels[n]), reverse=True)
    st.sidebar.caption(f"Running now ({len(running)})")
    with st.sidebar:
        for n in newest_first:
            if n in running:
                toggle(n)
        finished = [n for n in newest_first if n not in running]
        with st.expander(f"Finished ({len(finished)})", expanded=False):
            for n in finished:
                toggle(n)
    return [n for n in labels if st.session_state[f"show::{n}"]]


def kpi_table(runs: dict) -> None:
    rows = []
    rerate = gather("rerate_sf.jsonl")
    for name, data in runs.items():
        tr, sp, lad = data["train"], data["selfplay"], data["ladder"]
        rr = rerated_points(data, rerate, "step")
        last_tr = tr.iloc[-1] if not tr.empty else {}
        last_sp = sp.iloc[-1] if not sp.empty else {}
        last_lad = lad.iloc[-1] if not lad.empty else {}
        games = last_sp.get("total_games", 0) or 0
        rows.append(
            {
                "run": name,
                "hours": round(float(last_sp.get("hours", last_tr.get("hours", 0)) or 0), 2),
                "step": int(last_tr.get("step", 0) or 0),
                "model": (
                    f"enc {int(last_tr.get('model_d_model') or 0)}x{int(last_tr.get('model_layers') or 0)}"
                    if last_tr.get("model_d_model") == last_tr.get("model_d_model")
                    and "model_d_model" in last_tr
                    and last_tr.get("model_d_model")
                    else f"{int(last_tr.get('model_blocks', 0) or 0)}x{int(last_tr.get('model_channels', 0) or 0)}"
                ),
                "params": f"{(last_tr.get('params', 0) or 0) / 1e6:.2f}M",
                "pos/s": round(float(last_sp.get("positions_per_sec", 0) or 0), 1),
                "games/s": round(float(last_sp.get("games_per_sec", 0) or 0), 2),
                "positions": f"{int(last_sp.get('total_records_written', 0) or 0):,}",
                "games": f"{int(games):,}",
                "avg plies": round((last_sp.get("total_game_plies", 0) or 0) / games, 1)
                if games
                else None,
                "ladder Elo": f"{last_lad.get('elo', float('nan')):.0f} ± {last_lad.get('elo_se', float('nan')):.0f}"
                if len(lad)
                else "–",
                "Elo · SF openings": f"{rr['elo'].iloc[-1]:.0f} @ {rr['step'].iloc[-1] / 1000:g}k"
                if len(rr)
                else "–",
                "puzzles": f"{last_lad.get('puzzles.line', float('nan')):.3f}" if len(lad) else "–",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def budget_panel() -> None:
    if not BUDGET.exists():
        st.caption("budget.json not found (chessrl-vast guard pushes it every 5 min)")
        return
    b = json.loads(BUDGET.read_text())
    c = st.columns(5)
    c[0].metric("Spent", f"${b['spent']:.2f}", f"of ${b['cap']:.0f}")
    c[1].metric("Burn", f"${b['burn_per_hour']:.2f}/h")
    c[2].metric("Hours left at burn", f"{b['hours_left']:.0f}" if b.get("hours_left") else "–")
    live = [i for i in b["instances"] if i["ended"] is None]
    c[3].metric("Boxes up", len(live))
    c[4].metric("Updated", f"{(time.time() - b['t']) / 60:.0f} min ago")
    st.progress(min(1.0, b["spent"] / b["cap"]))


# The first post's best automatic match: 55% against Stockfish-1400 (800 simulations, 50 games).
FIRST_POST_BEST = 1400 + 400 * np.log10(0.55 / 0.45)


def network_label(engine: str) -> str:
    # <date>_<time>_<hex>_<run name>_step_<NNNNNNNN> -> "<run name> @ <k steps>k"
    parts = engine.split("_")
    if len(parts) > 5 and parts[-2] == "step":
        return f"{'_'.join(parts[3:-2])} @ {int(parts[-1]) / 1000:g}k"
    return engine


def stockfish_rating(rows: pd.DataFrame) -> tuple[float, float] | None:
    """One rating on Stockfish's UCI_Elo scale from the scores against every level (maximum
    likelihood, draws as half points, as in the ladder fit), with its standard error. None when
    the scores do not pin it down (all won or all lost)."""
    grid = np.arange(0.0, 4000.0)
    ll, info = np.zeros_like(grid), np.zeros_like(grid)
    for _, r in rows.iterrows():
        p = np.clip(1 / (1 + 10 ** ((r["sf_elo"] - grid) / 400)), 1e-12, 1 - 1e-12)
        s = r["w"] + 0.5 * r["d"]
        ll += s * np.log(p) + (r["games"] - s) * np.log(1 - p)
        info += r["games"] * p * (1 - p) * (np.log(10) / 400) ** 2
    i = int(np.argmax(ll))
    if i in (0, len(grid) - 1):
        return None
    return float(grid[i]), float(1 / np.sqrt(info[i]))


def benchmarks_panel() -> None:
    """Re-ratings on the Stockfish openings and the first post's Stockfish benchmark, wherever
    they were written (any run's metrics/rerate_sf.jsonl and metrics/sf_benchmark.jsonl)."""
    rerate, bench = gather("rerate_sf.jsonl"), gather("sf_benchmark.jsonl")
    if rerate.empty and bench.empty:
        return
    with st.expander(
        "Re-ratings on Stockfish openings · first-post Stockfish benchmark", expanded=True
    ):
        if not rerate.empty:
            st.caption(
                "Checkpoints re-rated on the balanced Stockfish openings (anchors recalibrated on them). "
                "'old Elo' is the run's rating on random openings: a different scale."
            )
            cols = [c for c in ("run", "step", "elo", "elo_se", "old_elo") if c in rerate]
            st.dataframe(
                rerate[cols]
                .drop_duplicates(["run", "step"], keep="last")
                .sort_values(["run", "step"]),
                hide_index=True,
                width="stretch",
            )
        ok = bench[bench["score"].notna()] if "score" in bench else pd.DataFrame()
        if not ok.empty:
            st.caption(
                "The first post's protocol: Stockfish 16 with UCI_Elo, 100 ms per move; the network "
                "searches 800 simulations per move; games from the initial position. 'Stockfish scale' "
                "is one rating fitted to the scores against every level, on Stockfish's UCI_Elo "
                "scale (not the ladder's)."
            )
            ok = ok.drop_duplicates(["engine", "sf_elo"], keep="last").assign(
                network=ok["engine"].map(network_label)
            )
            rows = []
            for net, g in ok.groupby("network"):
                row = {"network": net}
                for _, r in g.sort_values("sf_elo").iterrows():
                    row[f"vs SF-{int(r['sf_elo'])}"] = (
                        f"{r['score']:.0%}  ({int(r['w'])}W {int(r['d'])}D {int(r['l'])}L)"
                    )
                fit = stockfish_rating(g)
                row["rating"], row["se"] = fit if fit else (None, None)
                rows.append(row)
            table = pd.DataFrame(rows).sort_values("rating", na_position="first")
            fitted = table[table["rating"].notna()]
            if not fitted.empty:
                fig = go.Figure(
                    go.Scatter(
                        x=fitted["rating"],
                        y=fitted["network"],
                        mode="markers",
                        marker=dict(size=10, color=PALETTE[0]),
                        error_x=dict(type="data", array=fitted["se"], visible=True, thickness=1.5),
                        customdata=fitted["se"],
                        hovertemplate="%{y}: %{x:,.0f} ± %{customdata:.0f}<extra></extra>",
                    )
                )
                fig.add_vline(
                    x=FIRST_POST_BEST,
                    line=dict(color="#64748b", dash="dash", width=1.5),
                    annotation_text=f"first post, best match (55% vs SF-1400): {FIRST_POST_BEST:,.0f}",
                    annotation_position="top left",
                )
                fig.update_layout(
                    title="Stockfish-scale Elo (first post's benchmark)",
                    height=120 + 36 * len(fitted),
                    margin=dict(l=10, r=10, t=60, b=10),
                    xaxis_title="Elo on Stockfish's UCI_Elo scale",
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")
            table["Stockfish scale"] = [
                f"{r:,.0f} ± {s:.0f}" if r == r and r is not None else "–"
                for r, s in zip(table["rating"], table["se"], strict=True)
            ]
            st.dataframe(
                table.drop(columns=["rating", "se"]).iloc[::-1], hide_index=True, width="stretch"
            )
        if "error" in bench and bench["error"].notna().any():
            st.caption(
                f"{int(bench['error'].notna().sum())} benchmark match(es) failed (see sf_benchmark.jsonl)."
            )


FIRST_MOVE_COLORS = {"e4": PALETTE[0], "d4": PALETTE[1], "Nf3": PALETTE[2], "c4": PALETTE[3]}
OTHER_COLORS = PALETTE[4:7]


def first_move_san(uci: str) -> str:
    # From the starting position the knights stand on b1 / g1; every other first move is a pawn's.
    return ("N" if uci[:2] in ("b1", "g1") else "") + uci[2:]


def opening_panel(runs: dict, xkey: str) -> None:
    """First-move preferences at the starting position (ladder_daemon's opening probe): the raw
    policy and a 400-sim search without noise, as stacked shares of the four main moves + the rest."""
    shown = [
        (n, d["ladder"])
        for n, d in runs.items()
        if not d["ladder"].empty
        and xkey in d["ladder"]
        and any(c.startswith("opening.policy.") for c in d["ladder"])
    ]
    if not shown:
        return
    st.subheader("Opening: first move from the starting position")
    st.caption(
        "Measured at each rating: the network's raw policy, and a 400-simulation search without noise. "
        "The four main moves of each chart, the rest grouped as 'other'."
    )
    for name, lad in shown:
        cols = st.columns(2)
        for col, kind, label in (
            (cols[0], "policy", "network alone"),
            (cols[1], "search", "after 400 simulations"),
        ):
            keys = [c for c in lad if c.startswith(f"opening.{kind}.")]
            part = lad.dropna(subset=keys[:1]).sort_values(xkey)
            shares = part[keys].fillna(0.0)
            top = list(shares.max().sort_values(ascending=False).index[:4])
            fig = go.Figure()
            spare = iter(OTHER_COLORS)
            for k in top:
                san = first_move_san(k.split(".")[-1])
                fig.add_trace(
                    go.Scatter(
                        x=part[xkey],
                        y=shares[k],
                        name=san,
                        stackgroup="one",
                        mode="lines",
                        line=dict(width=0.5, color=FIRST_MOVE_COLORS.get(san) or next(spare)),
                        hovertemplate=f"{san}: %{{y:.0%}}<extra></extra>",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=part[xkey],
                    y=(1 - shares[top].sum(axis=1)).clip(lower=0),
                    name="other",
                    stackgroup="one",
                    mode="lines",
                    line=dict(width=0.5, color="#94a3b8"),
                    hovertemplate="other: %{y:.0%}<extra></extra>",
                )
            )
            fig.update_layout(
                title=f"{name} · {label}",
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title=XLABEL[xkey],
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                legend=dict(orientation="h", y=-0.3),
            )
            col.plotly_chart(fig, width="stretch")


def selfplay_derived(runs: dict) -> None:
    """Per-window rates from the cumulative self-play counters."""
    for data in runs.values():
        sp = data["selfplay"]
        if sp.empty:
            continue
        d = sp.filter(like="total_").diff()
        games = d.get("total_games")
        if games is None:
            continue
        g = games.where(games > 0)
        sp["game_length"] = d["total_game_plies"] / g
        sp["frac_decisive"] = (d["total_white_wins"] + d["total_black_wins"]) / g
        sp["frac_checkmate"] = d["total_end_checkmate"] / g
        sp["frac_resign"] = d["total_end_resign"] / g
        sp["frac_max_plies"] = d["total_end_max_plies"] / g


@st.fragment(run_every=60)
def body() -> None:
    all_runs = list_runs()
    st.sidebar.header("Runs")
    if not all_runs:
        st.info(f"No runs with metrics under {', '.join(map(str, RUNS_ROOTS))}")
        return
    labels = {run_label(p): p for p in all_runs}
    chosen = select_runs(labels)
    if not chosen:
        st.info("No runs selected: tick some in the sidebar.")
    xkey = st.sidebar.radio("x axis", list(XLABEL), format_func=XLABEL.get)
    scale = st.sidebar.radio(
        "ladder scale",
        ("both", "random", "sf"),
        format_func={
            "both": "both",
            "random": "random openings (original)",
            "sf": "Stockfish openings",
        }.get,
    )
    k = st.sidebar.slider("smoothing (points)", 1, 200, 20)
    runs = {n: load_run(str(labels[n])) for n in chosen}
    selfplay_derived(runs)

    st.title("♟ ChessRL")
    budget_panel()
    kpi_table(runs)

    st.subheader("Strength")
    c = st.columns(2)
    with c[0]:
        fig = go.Figure()
        any_sf = False
        rerate = gather("rerate_sf.jsonl")
        for i, (name, data) in enumerate(runs.items()):
            lad = data["ladder"]
            lad = lad if not lad.empty and xkey in lad else pd.DataFrame()
            # Two opening sets, two scales: rows without the field predate the switch (random).
            sets = (
                lad["openings"].fillna("random")
                if "openings" in lad
                else pd.Series("random", index=lad.index)
            )
            # The Stockfish-openings scale: live ladder rows plus checkpoints re-rated afterwards.
            sf = pd.concat([lad[sets == "sf"], rerated_points(data, rerate, xkey)])
            by_set = {
                "random": lad[sets == "random"],
                "sf": sf.sort_values(xkey) if not sf.empty else sf,
            }
            for oset, dash, suffix in (
                ("random", "solid", ""),
                ("sf", "dash", " · Stockfish openings"),
            ):
                part = by_set[oset]
                if part.empty or scale not in ("both", oset):
                    continue
                any_sf |= oset == "sf"
                fig.add_trace(
                    go.Scatter(
                        x=part[xkey],
                        y=part["elo"],
                        mode="lines+markers",
                        name=name + suffix,
                        error_y=dict(type="data", array=part["elo_se"], visible=True, thickness=1),
                        line=dict(color=PALETTE[i % len(PALETTE)], width=2, dash=dash),
                    )
                )
        fig.update_layout(
            title="Anchor-ladder Elo (random mover = 0)",
            height=340,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title=XLABEL[xkey],
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, width="stretch")
        if any_sf:
            st.warning(
                "Two rating scales. Solid: games from 8 random plies (the original ladder). Dashed: "
                "Stockfish's balanced openings, with anchors recalibrated on them (from 2026-10-26; "
                "earlier runs re-rated every 10k steps). "
                "They are not comparable: the balanced-opening scale is stretched at the top "
                "(Stockfish skill 20 at 1k nodes: 2,068 on random openings, 2,517 on balanced ones)."
            )
    with c[1]:
        line_chart(
            "Puzzle accuracy (1k Lichess, strict line, MCTS-100)",
            runs,
            "ladder",
            ["puzzles.line", "puzzles.first_move"],
            xkey,
            1,
            yfmt=".0%",
            height=340,
        )

    benchmarks_panel()
    opening_panel(runs, xkey)

    st.subheader("Learning")
    c = st.columns(3)
    with c[0]:
        line_chart(
            "Policy loss (CE vs visit distribution)", runs, "train", ["loss/policy_visits"], xkey, k
        )
    with c[1]:
        line_chart("KL(search ‖ policy)", runs, "train", ["policy/kl_to_visits"], xkey, k)
    with c[2]:
        line_chart(
            "Value losses (outcome · search)",
            runs,
            "train",
            ["loss/value", "loss/value_q"],
            xkey,
            k,
        )
    c = st.columns(3)
    with c[0]:
        line_chart(
            "Policy top-1 = search's top move",
            runs,
            "train",
            ["acc/policy_top_visit"],
            xkey,
            k,
            yfmt=".0%",
        )
    with c[1]:
        line_chart(
            "Entropy (policy · visit target)",
            runs,
            "train",
            ["entropy/policy", "entropy/visit_target"],
            xkey,
            k,
        )
    with c[2]:
        line_chart("Learning rate", runs, "train", ["lr"], xkey, 1, log_y=True)

    st.subheader("Signal quality")
    c = st.columns(3)
    with c[0]:
        line_chart(
            "Decisive game outcomes in batches",
            runs,
            "train",
            ["buffer/decisive_outcome"],
            xkey,
            k,
            yfmt=".0%",
        )
    with c[1]:
        line_chart(
            "Outcome vs search WDL agreement",
            runs,
            "train",
            ["buffer/wdl_agreement"],
            xkey,
            k,
            yfmt=".0%",
        )
    with c[2]:
        line_chart(
            "Castle rate (rare-action canary)",
            runs,
            "train",
            ["buffer/castle_rate"],
            xkey,
            k,
            yfmt=".2%",
        )
    c = st.columns(3)
    with c[0]:
        line_chart("Game length (plies)", runs, "selfplay", ["game_length"], xkey, 5)
    with c[1]:
        line_chart(
            "How games end",
            runs,
            "selfplay",
            ["frac_checkmate", "frac_resign", "frac_max_plies"],
            xkey,
            5,
            yfmt=".0%",
        )
    with c[2]:
        line_chart(
            "Near-uniform search WDL targets",
            runs,
            "train",
            ["buffer/frac_near_uniform"],
            xkey,
            k,
            yfmt=".0%",
        )

    st.subheader("Throughput")
    c = st.columns(3)
    with c[0]:
        line_chart("Self-play positions / s", runs, "selfplay", ["positions_per_sec"], xkey, 3)
    with c[1]:
        line_chart("Self-play games / s", runs, "selfplay", ["games_per_sec"], xkey, 3)
    with c[2]:
        line_chart("Samples per record (target_k)", runs, "train", ["pacing/k_actual"], xkey, k)
    c = st.columns(3)
    with c[0]:
        line_chart("Trainer step time (s)", runs, "train", ["pacing/step_time_s"], xkey, k)
    with c[1]:
        line_chart("Sampling window (records)", runs, "train", ["buffer/window"], xkey, 1)
    with c[2]:
        line_chart("MCTS nodes / s", runs, "selfplay", ["nodes_per_sec"], xkey, 3)

    with st.expander("Configs"):
        for name, data in runs.items():
            st.markdown(f"**{name}**")
            st.json(data["config"], expanded=False)


body()
