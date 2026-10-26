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
RUNS_ROOTS = [Path(p) for p in os.environ.get("CHESSRL_RUNS_ROOT", str(ROOT / "artifacts" / "runs")).split(":")]
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
        lad = lad[lad["t"] >= own_t0 - 1800].copy()  # a run's first rating may precede its metrics a little
    t0 = float(branch.get("parent_t0", own_t0))
    # Hours of training, not of wall clock: a run stopped and resumed later (or down
    # after a crash) would otherwise jump right by its downtime. Self-play logs every
    # 30 s; any longer silence counts as downtime.
    ts = np.sort(selfplay["t"].to_numpy(dtype=float)) if not selfplay.empty and "t" in selfplay else np.array([])
    gaps = np.diff(ts)
    big = np.append(gaps > 600, True)  # the open interval after the last row counts once it is > 600 s
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
        ys = ys + np.concatenate([[0.0], np.cumsum(resets)]) + float(branch.get("parent_positions", 0))
        for df in (train, events, lad):
            if not df.empty and "t" in df:
                df["positions"] = np.interp(df["t"].to_numpy(dtype=float), xs, ys)
        selfplay["positions"] = ys
    cfg = {}
    if (d / "config.yaml").exists():
        cfg = yaml.safe_load((d / "config.yaml").read_text()) or {}
    return {"train": train, "selfplay": selfplay, "events": events, "ladder": lad, "config": cfg, "t0": t0}


def list_runs() -> list[Path]:
    runs = [p for root in RUNS_ROOTS if root.exists() for p in root.rglob("metrics") if p.is_dir()]
    return sorted({p.parent for p in runs}, key=lambda p: p.name)  # names start with the start time


def smooth(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=1).mean() if k > 1 else s


def line_chart(title: str, runs: dict, table: str, cols: list[str], xkey: str, k: int,
               yfmt: str | None = None, height: int = 300, log_y: bool = False) -> None:
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
            fig.add_trace(go.Scatter(x=df[xkey], y=y, mode="lines", name=label,
                                     line=dict(color=PALETTE[i % len(PALETTE)], dash=dashes[j % 4], width=2)))
        ev = data["events"]
        if not ev.empty and xkey in ev:
            for _, e in ev[ev["event"] == "grow"].iterrows():
                fig.add_vline(x=e[xkey], line=dict(color=PALETTE[i % len(PALETTE)], dash="dot", width=1))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.25), xaxis_title=XLABEL[xkey])
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


# Runs shown by default: everything still running, plus these run names
# (comma-separated, e.g. "enc-deep,signal"; overridable with $CHESSRL_PINNED).
PINNED = {n.strip() for n in os.environ.get("CHESSRL_PINNED", "enc-deep").split(",") if n.strip()}
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
    for name, data in runs.items():
        tr, sp, lad = data["train"], data["selfplay"], data["ladder"]
        last_tr = tr.iloc[-1] if not tr.empty else {}
        last_sp = sp.iloc[-1] if not sp.empty else {}
        last_lad = lad.iloc[-1] if not lad.empty else {}
        games = last_sp.get("total_games", 0) or 0
        rows.append({
            "run": name,
            "hours": round(float(last_sp.get("hours", last_tr.get("hours", 0)) or 0), 2),
            "step": int(last_tr.get("step", 0) or 0),
            "model": (f"enc {int(last_tr.get('model_d_model') or 0)}x{int(last_tr.get('model_layers') or 0)}"
                      if last_tr.get("model_d_model") == last_tr.get("model_d_model") and "model_d_model" in last_tr
                      and last_tr.get("model_d_model")
                      else f"{int(last_tr.get('model_blocks', 0) or 0)}x{int(last_tr.get('model_channels', 0) or 0)}"),
            "params": f"{(last_tr.get('params', 0) or 0) / 1e6:.2f}M",
            "pos/s": round(float(last_sp.get("positions_per_sec", 0) or 0), 1),
            "games/s": round(float(last_sp.get("games_per_sec", 0) or 0), 2),
            "positions": f"{int(last_sp.get('total_records_written', 0) or 0):,}",
            "games": f"{int(games):,}",
            "avg plies": round((last_sp.get("total_game_plies", 0) or 0) / games, 1) if games else None,
            "ladder Elo": f"{last_lad.get('elo', float('nan')):.0f} ± {last_lad.get('elo_se', float('nan')):.0f}"
            if len(lad) else "–",
            "puzzles": f"{last_lad.get('puzzles.line', float('nan')):.3f}" if len(lad) else "–",
        })
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


def benchmarks_panel() -> None:
    """Re-ratings on the Stockfish openings and the first post's Stockfish benchmark, wherever
    they were written (any run's metrics/rerate_sf.jsonl and metrics/sf_benchmark.jsonl)."""
    def gather(name: str) -> pd.DataFrame:
        # <root>/<run>/metrics on a box, <root>/<box>/<run>/metrics in the synced copy at home.
        files = [p for root in RUNS_ROOTS for pat in (f"*/metrics/{name}", f"*/*/metrics/{name}") for p in root.glob(pat)]
        return pd.concat([read_jsonl(p) for p in files] or [pd.DataFrame()], ignore_index=True)

    rerate, bench = gather("rerate_sf.jsonl"), gather("sf_benchmark.jsonl")
    if rerate.empty and bench.empty:
        return
    with st.expander("Re-ratings on Stockfish openings · first-post Stockfish benchmark", expanded=True):
        if not rerate.empty:
            st.caption("Checkpoints re-rated on the balanced Stockfish openings (anchors recalibrated on them). "
                       "'old Elo' is the run's rating on random openings: a different scale.")
            cols = [c for c in ("run", "step", "elo", "elo_se", "old_elo") if c in rerate]
            st.dataframe(rerate[cols].drop_duplicates(["run", "step"], keep="last").sort_values(["run", "step"]),
                         hide_index=True, width="stretch")
        if not bench.empty:
            st.caption("The first post's protocol: Stockfish 16 with UCI_Elo, 100 ms per move; the network "
                       "searches 800 simulations per move; games from the initial position.")
            cols = [c for c in ("engine", "sf_elo", "games", "w", "d", "l", "score", "implied_elo") if c in bench]
            st.dataframe(bench[cols].sort_values(["engine", "sf_elo"]), hide_index=True, width="stretch")


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
        for i, (name, data) in enumerate(runs.items()):
            lad = data["ladder"]
            if lad.empty or xkey not in lad:
                continue
            # Two opening sets, two scales: rows without the field predate the switch (random).
            sets = lad["openings"].fillna("random") if "openings" in lad else pd.Series("random", index=lad.index)
            for oset, dash, suffix in (("random", "solid", ""), ("sf", "dash", " · Stockfish openings")):
                part = lad[sets == oset]
                if part.empty:
                    continue
                any_sf |= oset == "sf"
                fig.add_trace(go.Scatter(
                    x=part[xkey], y=part["elo"], mode="lines+markers", name=name + suffix,
                    error_y=dict(type="data", array=part["elo_se"], visible=True, thickness=1),
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2, dash=dash)))
        fig.update_layout(title="Anchor-ladder Elo (random mover = 0)", height=340,
                          margin=dict(l=10, r=10, t=40, b=10), xaxis_title=XLABEL[xkey],
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        if any_sf:
            st.warning("Two rating scales. Solid: games from 8 random plies (the original ladder). Dashed: "
                       "Stockfish's balanced openings, with anchors recalibrated on them (from 2026-10-26). "
                       "They are not comparable: the balanced-opening scale is stretched at the top "
                       "(Stockfish skill 20 at 1k nodes: 2,068 on random openings, 2,517 on balanced ones).")
    with c[1]:
        line_chart("Puzzle accuracy (1k Lichess, strict line, MCTS-100)", runs, "ladder",
                   ["puzzles.line", "puzzles.first_move"], xkey, 1, yfmt=".0%", height=340)

    benchmarks_panel()

    st.subheader("Learning")
    c = st.columns(3)
    with c[0]:
        line_chart("Policy loss (CE vs visit distribution)", runs, "train", ["loss/policy_visits"], xkey, k)
    with c[1]:
        line_chart("KL(search ‖ policy)", runs, "train", ["policy/kl_to_visits"], xkey, k)
    with c[2]:
        line_chart("Value losses (outcome · search)", runs, "train", ["loss/value", "loss/value_q"], xkey, k)
    c = st.columns(3)
    with c[0]:
        line_chart("Policy top-1 = search's top move", runs, "train", ["acc/policy_top_visit"], xkey, k, yfmt=".0%")
    with c[1]:
        line_chart("Entropy (policy · visit target)", runs, "train",
                   ["entropy/policy", "entropy/visit_target"], xkey, k)
    with c[2]:
        line_chart("Learning rate", runs, "train", ["lr"], xkey, 1, log_y=True)

    st.subheader("Signal quality")
    c = st.columns(3)
    with c[0]:
        line_chart("Decisive game outcomes in batches", runs, "train", ["buffer/decisive_outcome"], xkey, k,
                   yfmt=".0%")
    with c[1]:
        line_chart("Outcome vs search WDL agreement", runs, "train", ["buffer/wdl_agreement"], xkey, k, yfmt=".0%")
    with c[2]:
        line_chart("Castle rate (rare-action canary)", runs, "train", ["buffer/castle_rate"], xkey, k, yfmt=".2%")
    c = st.columns(3)
    with c[0]:
        line_chart("Game length (plies)", runs, "selfplay", ["game_length"], xkey, 5)
    with c[1]:
        line_chart("How games end", runs, "selfplay", ["frac_checkmate", "frac_resign", "frac_max_plies"],
                   xkey, 5, yfmt=".0%")
    with c[2]:
        line_chart("Near-uniform search WDL targets", runs, "train", ["buffer/frac_near_uniform"], xkey, k,
                   yfmt=".0%")

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
