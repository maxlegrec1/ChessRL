"""Charts for the "next round" blog post, from runs synced under ~/chessrl-runs.

    .venv/bin/python docs/blog/make_charts.py      # writes docs/blog/img/*.png

Palette: the dataviz reference categorical slots (validated: slots 1-4 adjacent;
blue/orange all-pairs). One colour per recipe across every chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path.home() / "chessrl-runs"
OUT = Path(__file__).resolve().parent / "img"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "savefig.dpi": 160,
    "savefig.facecolor": SURFACE,
})


def jl(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        return []


def run_dir(box: str, suffix: str) -> Path:
    return sorted((RUNS / box).glob(f"*_{suffix}"))[-1]


def axes(d: Path):
    """(t0, active_hours(t), records_at(t), trainer rows, self-play rows) for a run directory."""
    tr = jl(d / "metrics/train.jsonl")
    sp = jl(d / "metrics/selfplay.jsonl")
    t0 = min(r["t"] for r in tr + sp)
    # Records written, cumulative across self-play restarts (the counter restarts at 0).
    cum, off, prev, pts = 0.0, 0.0, 0.0, []
    for r in sp:
        v = float(r.get("total_records_written", 0))
        if v < prev:
            off += prev
        prev = v
        cum = v + off
        pts.append((r["t"], cum))

    def records_at(t: float) -> float:
        before = [c for tt, c in pts if tt <= t]
        return before[-1] if before else 0.0

    # Hours of training, not wall clock (same rule as the dashboard): self-play
    # logs every 30 s, so a longer silence (a stop/resume, a crash) is downtime.
    ts = sorted(r["t"] for r in sp)

    def active_hours(t: float) -> float:
        down, prev = 0.0, None
        for tt in ts:
            if tt > t:
                if prev is not None and tt - prev > 600:
                    down += max(0.0, t - prev - 30)
                break
            if prev is not None and tt - prev > 600:
                down += tt - prev - 30
            prev = tt
        else:
            if prev is not None and t - prev > 600:
                down += t - prev - 30
        return max(0.0, (t - t0 - down) / 3600)

    return t0, active_hours, records_at, tr, sp


def load(box: str, suffix: str, max_step: int | None = None) -> dict:
    """The ladder as it ran (every 30 min, random 8-ply openings): Elo and puzzle accuracy on
    hours-since-start and cumulative self-play records."""
    d = run_dir(box, suffix)
    t0, active_hours, records_at, _, _ = axes(d)
    lad = jl(d / "ladder/ladder.jsonl")
    lad = [r for r in lad if r["t"] >= t0 - 1800 and (max_step is None or r["step"] <= max_step)]
    return {
        "hours": [active_hours(r["t"]) for r in lad],
        "records_m": [records_at(r["t"]) / 1e6 for r in lad],
        "steps_k": [r["step"] / 1000 for r in lad],
        "elo": [r["elo"] for r in lad],
        "puzzles": [(r.get("puzzles") or {}).get("line") for r in lad],
    }


def rerated(box: str, suffix: str, max_step: int | None = None) -> dict:
    """The run's kept checkpoints (one per 10k steps) re-rated on Stockfish's balanced openings
    (scripts/rerate.py; rows may come from another box), on the same axes as load(). The first
    point is the untrained network, rated by the ladder before training: below ~800 Elo the two
    opening sets give the same scale. A checkpoint's time is that of the first trainer row at or
    after its step (the original code wrote no trainer log: its first self-play row with that
    engine)."""
    d = run_dir(box, suffix)
    _, active_hours, records_at, tr, sp = axes(d)
    rows = {}
    for f in sorted(RUNS.glob("*/*/metrics/rerate_sf.jsonl")):
        for r in jl(f):
            if r.get("run") == d.name and r.get("openings") == "sf" and (max_step is None or r["step"] <= max_step):
                rows[r["step"]] = r  # a step rated twice: the last row wins
    steps = [(r["step"], r["t"]) for r in tr if "step" in r and "t" in r and "event" not in r]
    if not steps:
        steps = [(r["engine_step"], r["t"]) for r in sp if "engine_step" in r]
    live = load(box, suffix)
    out = {"hours": [0.0], "records_m": [0.0], "steps_k": [0.0], "elo": [live["elo"][0]]}
    for step in sorted(rows):
        t = next((t for s, t in steps if s >= step), steps[-1][1])
        out["hours"].append(active_hours(t))
        out["records_m"].append(records_at(t) / 1e6)
        out["steps_k"].append(step / 1000)
        out["elo"].append(rows[step]["elo"])
    return out


def end_label(ax, x, y, text, leader=None):
    """End dot + value. leader=(dx, dy) in points moves the label off a crowded end, with a thin connector."""
    ax.plot([x], [y], "o", ms=6, color=ax.lines[-1].get_color(), mec=SURFACE, mew=2, zorder=5)
    if leader:
        ax.annotate(text, (x, y), xytext=leader, textcoords="offset points", va="center", color=INK2, fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=4))
    else:
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", color=INK2, fontsize=10)


def chart_speedup() -> None:
    series = [
        ("Original code (main)", YELLOW, load("armA", "baseline")),
        ("+ learning-signal fixes", AQUA, load("dev1", "signal")),
        ("+ small transformer encoder", ORANGE, load("armE", "encoder", max_step=40000)),
        ("+ half the search per move", BLUE, load("armA", "enc-sims400")),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name, color, s in series:
        ax.plot(s["hours"], s["elo"], color=color, label=name)
        end_label(ax, s["hours"][-1], s["elo"][-1], f"{s['elo'][-1]:,.0f}", leader=(18, -34) if color == BLUE else None)
    ax.set_xlim(0, 23.5)
    ax.set_ylim(-100, 2100)
    ax.set_xlabel("hours of training (one RTX 4090 each)")
    ax.set_ylabel("ladder Elo (random play = 0)")
    ax.set_title("Same GPU, four recipes: Elo against calibrated Stockfish anchors", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=10, borderaxespad=0.2)
    fig.text(0.01, 0.01, "Each rating: 96 games at 100 simulations per move, ±40 Elo. "
             "Encoder curve stops at 17 h, where its inference engine started to diverge (see text).",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "speedup_elo_hours.png")
    plt.close(fig)


def chart_search_budget() -> None:
    e = load("armE", "encoder", max_step=40000)
    s2 = load("armF", "encoder-s2")
    s4 = load("armA", "enc-sims400")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, key, xlabel, xmax in ((a, "hours", "hours of training", 8.6),
                                  (b, "records_m", "self-play positions recorded (millions)", 12)):
        for name, color, s in (("800 sims, seed 1", ORANGE, e), ("800 sims, seed 2", ORANGE, s2),
                               ("400 sims", BLUE, s4)):
            xs = [x for x in s[key] if x <= xmax]
            ax.plot(xs, s["elo"][: len(xs)], color=color, alpha=1.0 if "seed 2" not in name else 0.55, label=name)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
    a.set_ylim(0, 2000)
    a.set_ylabel("ladder Elo")
    a.set_title("Per hour, 400 sims gets there first")
    b.set_title("Per position: behind mid-run, level by 8M")
    a.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "search_budget.png")
    plt.close(fig)


def chart_fp16() -> None:
    # Measured with artifacts/scratch/act_scale.py (fp32, 512 puzzle positions): max |x|
    # entering the final RMSNorm. No decay: the encoder arm (E); decay: sims-400 + AdamW 0.05.
    VIOLET = "#6250d6"  # validated against ORANGE (all-pairs, light)
    series = [("no weight decay", ORANGE, [10_000, 20_000, 30_000, 40_000, 45_500], [46.0, 101.3, 194.0, 280.2, 349.3]),
              ("weight decay 0.05", VIOLET, [10_000, 20_000, 30_000, 50_000, 70_000], [24.9, 35.7, 45.8, 64.4, 79.9])]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, color, steps, vals in series:
        xs = [s / 1000 for s in steps]
        ax.plot(xs, vals, color=color, label=name)
        ax.plot(xs, vals, "o", ms=6, color=color, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"{vals[-1]:.0f}", (xs[-1], vals[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2)
    ax.axhline(256, color=INK2, lw=1)
    ax.annotate("x² overflows FP16 above |x| = 256", (10.5, 256), xytext=(0, 6), textcoords="offset points",
                color=INK2, fontsize=10)
    ax.set_xlim(8, 76)
    ax.set_ylim(0, 400)
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("largest activation into the final norm")
    ax.set_title("Without weight decay the residual stream keeps growing")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fp16_activation_growth.png")
    plt.close(fig)


def chart_ablations() -> None:
    # Ladder Elo at 6M self-play positions (docs/NEXT_ROUND.md, "Ablations at 6 h").
    rows = [("All learning-signal fixes", 1198), ("without the value-head fix", 1137),
            ("without opening-only temperature", 1130), ("without visit-distribution policy targets", 466)]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    bars = ax.barh(names, vals, height=0.5, color=AQUA)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:,}", (v, bar.get_y() + bar.get_height() / 2), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2)
    ax.set_xlim(0, 1400)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ladder Elo at 6M self-play positions (seed noise ≈ ±100)")
    ax.set_title("Which fix mattered: remove one at a time")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png")
    plt.close(fig)


CHART_JSON = Path(__file__).resolve().parent / "site" / "components" / "charts" / "chessrl-next-round.json"


def export_json() -> None:
    """The same data for the blog's native SVG charts (docs/blog/site/components/charts/).

    Elo: the kept checkpoints re-rated on the balanced openings (rerated()). Puzzles: the ladder
    as it ran (they do not depend on the ladder's openings)."""
    pts = lambda s, key="hours", y="elo": [[round(x, 2), round(v, 3 if y == "puzzles" else None)]  # noqa: E731
                                           for x, v in zip(s[key], s[y]) if v is not None]
    recipes = [("Original code (first post)", 4, "armA", "baseline", None),
               ("+ better training targets", 3, "dev1", "signal", None),
               # Its later steps ran on the broken FP16 engine (see the FP16 section).
               ("+ small transformer", 2, "armE", "encoder", 40000),
               ("+ half the search per move", 1, "armA", "enc-sims400", 60000)]
    enc, s400 = rerated("armE", "encoder", 40000), rerated("armA", "enc-sims400", 60000)

    def at(box: str, suffix: str, step_k: float) -> int:
        s = rerated(box, suffix)
        return round(s["elo"][s["steps_k"].index(step_k)])

    data = {
        "speedup": [{"name": n, "slot": k, "dots": True, "points": pts(rerated(b, s, m))} for n, k, b, s, m in recipes],
        "puzzles": [{"name": n, "slot": k, "points": pts(load(b, s, m), y="puzzles")} for n, k, b, s, m in recipes],
        "searchBudget": [
            {"name": "800 simulations, seed 1", "slot": 2, "dots": True, "points": pts(enc)},
            {"name": "800 simulations, seed 2", "slot": 2, "dim": True, "dots": True,
             "points": pts(rerated("armF", "encoder-s2"))},
            {"name": "400 simulations", "slot": 1, "dots": True, "points": pts(s400)},
        ],
        # Measured with artifacts/scratch/act_scale.py (fp32, 512 puzzle positions).
        "activation": [
            {"name": "no weight decay", "slot": 2,
             "points": [[10, 46.0], [20, 101.3], [30, 194.0], [40, 280.2], [45.5, 349.3]]},
            {"name": "weight decay 0.05", "slot": 7,
             "points": [[10, 24.9], [20, 35.7], [30, 45.8], [50, 64.4], [70, 79.9]]},
        ],
        # After 20k training steps (the ablation runs kept checkpoints at 10k and 20k only).
        "ablations": [
            {"label": "All three changes", "value": at("dev1", "signal", 20)},
            {"label": "without the value-head change", "value": at("abl-value", "ablate-value", 20)},
            {"label": "without limiting randomness to the opening", "value": at("abl-temp", "ablate-temp", 20)},
            {"label": "without the visit-count policy target", "value": at("abl-policy", "ablate-policy", 20)},
        ],
    }
    CHART_JSON.parent.mkdir(parents=True, exist_ok=True)
    CHART_JSON.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    # The post's charts are the native ones (export_json); the PNGs (first draft, the ladder as it
    # ran) are only written with --png.
    if "--png" in __import__("sys").argv:
        OUT.mkdir(parents=True, exist_ok=True)
        chart_speedup()
        chart_search_budget()
        chart_fp16()
        chart_ablations()
    export_json()
    print("wrote", CHART_JSON.name)
