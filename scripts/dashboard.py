"""ChessRL training dashboard.

Single-page Streamlit dashboard surfacing three live metrics:

1. Win rate vs Stockfish-1400 (from autobrawl daemon's state.json).
2. Elo chain — per-5k-step ΔElo + cumulative (from model_chain_brawl CSVs).
3. Castle rate (from W&B history, if accessible; rare-action learning signal).

Run:
    uv run streamlit run scripts/dashboard.py
Then open http://localhost:8501.

Auto-refreshes once per minute. Pure read-only — never modifies training,
the autobrawl daemon, or any artifact under artifacts/runs/.
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
AUTOBRAWL_STATE = ROOT / "artifacts/exp/sf1400_autobrawl/state.json"
ELO_CHAIN_GLOB = sorted(ROOT.glob("artifacts/exp/elo_chain*/results.csv"))
DEFAULT_RUN_DIR = ROOT / "artifacts/runs"
SF_ELO_ANCHOR = 1400

st.set_page_config(page_title="ChessRL dashboard", page_icon="♟", layout="wide")


# ──────────────────────────────────────────────────────────────────────────── #
# Data loaders (cached so refresh is cheap)
# ──────────────────────────────────────────────────────────────────────────── #
_RESULT_RE = re.compile(r"(\d+)W\s*/\s*(\d+)D\s*/\s*(\d+)L\s*->\s*(\d+\.\d+)/(\d+)\s*\(([\d.]+)%\)")


@st.cache_data(ttl=60)
def load_autobrawl() -> list[dict]:
    """Returns sorted list of {step, w, d, l, pct, elo} dicts."""
    if not AUTOBRAWL_STATE.is_file():
        return []
    d = json.loads(AUTOBRAWL_STATE.read_text())
    results = d.get("results", {})
    out = []
    for s_str, line in results.items():
        m = _RESULT_RE.search(line)
        if not m:
            continue
        w, dd, l, _score, n, pct = m.groups()
        p = float(pct) / 100.0
        if 0.0 < p < 1.0:
            elo = SF_ELO_ANCHOR + (-400.0 * math.log10((1 - p) / p))
        else:
            elo = SF_ELO_ANCHOR + (800.0 if p >= 1.0 else -800.0)
        out.append(
            dict(step=int(s_str), w=int(w), d=int(dd), l=int(l), n=int(n), pct=float(pct), elo=elo)
        )
    out.sort(key=lambda r: r["step"])
    return out


@st.cache_data(ttl=60)
def load_elo_chain() -> list[dict]:
    """Merge all elo_chain*/results.csv files into one sorted list with
    cumulative Elo computed in time order. Sign convention: CSV delta_elo is
    (older - newer); cumulative gain is therefore -sum(delta_elo)."""
    seen = set()
    rows = []
    for path in ELO_CHAIN_GLOB:
        with path.open() as fh:
            for r in csv.DictReader(fh):
                key = (int(r["from_step"]), int(r["to_step"]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
    rows.sort(key=lambda r: int(r["from_step"]))
    out = []
    cum = 0.0
    for r in rows:
        gain = -float(r["delta_elo"])
        cum += gain
        out.append(
            dict(
                from_step=int(r["from_step"]),
                to_step=int(r["to_step"]),
                delta_elo=gain,
                cumulative=cum,
            )
        )
    return out


@st.cache_data(ttl=60)
def load_castle_rate() -> tuple[list[int], list[float], str]:
    """Pull buffer/castle_rate history from W&B for the active training run.

    Returns (steps, rates, status_msg). Empty lists + message on failure.
    """
    rid_file = _find_wandb_run_id()
    if rid_file is None:
        return [], [], "no wandb_run_id.txt found in any artifacts/runs/<id>/"
    try:
        rid = rid_file.read_text().strip()
        import wandb

        api = wandb.Api()
        # Project is hardcoded to "self_play" to match trainer config default.
        run = api.run(f"self_play/{rid}")
        hist = run.history(samples=2000, keys=["buffer/castle_rate", "_step"])
        if hist is None or len(hist) == 0:
            return [], [], "W&B run has no buffer/castle_rate history yet"
        steps = list(hist["_step"])
        rates = list(hist["buffer/castle_rate"])
        return steps, rates, "OK"
    except Exception as exc:
        return [], [], f"W&B fetch failed: {exc}"


def _find_wandb_run_id() -> Path | None:
    """Return the wandb_run_id.txt of the most-recently-modified run dir."""
    if not DEFAULT_RUN_DIR.is_dir():
        return None
    candidates = sorted(
        DEFAULT_RUN_DIR.glob("*/wandb_run_id.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@st.cache_data(ttl=60)
def find_active_run() -> dict:
    """Discover the latest training run directory + its start time."""
    if not DEFAULT_RUN_DIR.is_dir():
        return {"name": None, "path": None, "started": None}
    runs = sorted(
        (p for p in DEFAULT_RUN_DIR.iterdir() if p.is_dir() and (p / "checkpoints").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return {"name": None, "path": None, "started": None}
    latest = runs[0]
    # Run id is "YYYYMMDD_HHMMSS_<hash>[_<name>]"
    parts = latest.name.split("_")
    started = None
    if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
        try:
            started = datetime.strptime(f"{parts[0]}{parts[1]}", "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            pass
    return {"name": latest.name, "path": latest, "started": started}


# ──────────────────────────────────────────────────────────────────────────── #
# UI
# ──────────────────────────────────────────────────────────────────────────── #
st.title("♟ ChessRL training dashboard")

run_info = find_active_run()
autobrawl = load_autobrawl()
chain = load_elo_chain()
castle_steps, castle_rates, castle_status = load_castle_rate()

# Header strip
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "active run",
    run_info["name"] if run_info["name"] else "—",
    help="latest directory under artifacts/runs/",
)
if run_info["started"] is not None:
    elapsed = datetime.now(UTC) - run_info["started"]
    hours = elapsed.total_seconds() / 3600
    c2.metric("running for", f"{hours:,.1f} h")
else:
    c2.metric("running for", "—")
if autobrawl:
    latest = autobrawl[-1]
    c3.metric(
        f"SF-{SF_ELO_ANCHOR} win % (step {latest['step'] // 1000}k)",
        f"{latest['pct']:.0f} %",
        delta=f"{latest['elo'] - SF_ELO_ANCHOR:+.0f} Elo",
    )
else:
    c3.metric(f"SF-{SF_ELO_ANCHOR} win %", "—")
if castle_rates:
    c4.metric(
        "castle rate (latest)",
        f"{castle_rates[-1]:.3%}",
        help="rare-action learning signal; canonical chess ≈ 0.8–1.2 %",
    )
else:
    c4.metric("castle rate", "—")

st.divider()

# ─── Section 1: SF-1400 win rate ────────────────────────────────────────────
st.subheader(f"Win rate vs Stockfish-{SF_ELO_ANCHOR}  (autobrawl daemon)")
if autobrawl:
    steps = [r["step"] for r in autobrawl]
    pcts = [r["pct"] for r in autobrawl]
    # 3-point rolling mean
    smooth = []
    for i in range(len(pcts)):
        lo = max(0, i - 1)
        hi = min(len(pcts), i + 2)
        smooth.append(sum(pcts[lo:hi]) / (hi - lo))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=pcts,
            mode="markers",
            name="brawl (n=50)",
            marker=dict(size=8, color="steelblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=smooth,
            mode="lines",
            name="3-brawl rolling mean",
            line=dict(color="darkblue", width=3),
        )
    )
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="red",
        annotation_text="parity with SF-1400",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="training step",
        yaxis_title=f"win % vs SF-{SF_ELO_ANCHOR} (sequential, no rand)",
        yaxis_range=[20, 70],
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{len(autobrawl)} brawls between step {min(steps) // 1000}k and "
        f"{max(steps) // 1000}k.  Range: {min(r['pct'] for r in autobrawl):.0f}% – "
        f"{max(r['pct'] for r in autobrawl):.0f}%."
    )
else:
    st.info(f"No data yet at {AUTOBRAWL_STATE}.")

st.divider()

# ─── Section 2: Elo chain ───────────────────────────────────────────────────
st.subheader("Elo chain   (consecutive-checkpoint head-to-head, 50 games/pair)")
if chain:
    labels = [f"{r['from_step'] // 1000}k→{r['to_step'] // 1000}k" for r in chain]
    deltas = [r["delta_elo"] for r in chain]
    cum_steps = [chain[0]["from_step"]] + [r["to_step"] for r in chain]
    cum = [0.0] + [r["cumulative"] for r in chain]

    col_left, col_right = st.columns([3, 4])
    with col_left:
        st.markdown("**Per-interval ΔElo**")
        fig1 = go.Figure(
            go.Bar(
                x=labels,
                y=deltas,
                marker_color=["seagreen" if d > 0 else "indianred" for d in deltas],
            )
        )
        fig1.update_layout(
            xaxis_title="",
            yaxis_title="ΔElo (newer − older)",
            height=380,
            margin=dict(l=40, r=20, t=20, b=80),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col_right:
        st.markdown("**Cumulative Elo trajectory**")
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=cum_steps,
                y=cum,
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(0,0,139,0.15)",
                line=dict(color="darkblue", width=2),
            )
        )
        for step, label, color in [
            (63500, "mirror-bug fix", "orange"),
            (140000, "Dirichlet bump", "purple"),
        ]:
            if cum_steps[0] <= step <= cum_steps[-1]:
                fig2.add_vline(
                    x=step,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=label,
                    annotation_position="top",
                )
        fig2.update_layout(
            xaxis_title="training step",
            yaxis_title=f"cumulative Elo (relative to {cum_steps[0] // 1000}k)",
            height=380,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        f"{len(chain)} intervals from step {cum_steps[0] // 1000}k to "
        f"{cum_steps[-1] // 1000}k.  Total cumulative gain: **{cum[-1]:+.0f} Elo**."
    )
else:
    st.info("No Elo-chain data yet.  Run `scripts/model_chain_brawl.py` to populate.")

st.divider()

# ─── Section 3: Castle rate ─────────────────────────────────────────────────
st.subheader("Castle rate   (rare-action learning signal, from W&B)")
if castle_rates:
    # Rolling mean (50-point) for smoothing
    win = 50
    smooth = []
    for i in range(len(castle_rates)):
        lo = max(0, i - win // 2)
        hi = min(len(castle_rates), i + win // 2 + 1)
        chunk = [x for x in castle_rates[lo:hi] if x is not None and not math.isnan(x)]
        smooth.append(sum(chunk) / max(1, len(chunk)))
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=castle_steps,
            y=castle_rates,
            mode="markers",
            marker=dict(size=3, color="lightblue"),
            name="per-batch",
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=castle_steps,
            y=smooth,
            mode="lines",
            line=dict(color="navy", width=2),
            name=f"{win}-point rolling mean",
        )
    )
    fig3.add_hrect(
        y0=0.008,
        y1=0.012,
        fillcolor="green",
        opacity=0.1,
        annotation_text="canonical chess (0.8–1.2%)",
        annotation_position="top left",
    )
    fig3.update_layout(
        xaxis_title="training step",
        yaxis_title="fraction of buffer rows that are castle UCIs",
        yaxis_tickformat=".2%",
        height=380,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        f"Latest: **{castle_rates[-1]:.3%}** (canonical chess is ~0.8–1.2 %). "
        f"Source: W&B run, key `buffer/castle_rate`."
    )
else:
    st.warning(f"Castle rate unavailable: {castle_status}")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.divider()
fc1, fc2 = st.columns([1, 6])
with fc1:
    if st.button("🔄 Regen PNG plots"):
        try:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "plot_autobrawl_trend.py")],
                check=True,
                capture_output=True,
                timeout=60,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "replot_elo_chain.py")],
                check=True,
                capture_output=True,
                timeout=60,
            )
            st.success("PNG plots regenerated under artifacts/exp/.")
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"plot regen failed: {exc}")
with fc2:
    st.caption(
        "Auto-refresh every 60 s. Data sources: "
        "`artifacts/exp/sf1400_autobrawl/state.json`, "
        "`artifacts/exp/elo_chain*/results.csv`, W&B run (castle rate). "
        "Dashboard is read-only — never modifies training state."
    )
