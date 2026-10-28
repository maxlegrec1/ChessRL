"""Chart data for the ChessRL follow-up post, and the README's chart images.

    .venv/bin/python docs/blog/make_charts.py            # post data, from the runs synced under ~/chessrl-runs
    .venv/bin/python docs/blog/make_charts.py --readme   # README images (docs/img/), from that data

The post draws its charts natively from site/components/charts/chessrl-next-round.json. The README
needs images, so --readme renders the same two runs with matplotlib, in a light and a dark version
(the README picks one with prefers-color-scheme). Series colours are the post's (charts.css),
validated for both surfaces; the end labels give the yellow line, below 3:1 on white, its relief.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path.home() / "chessrl-runs"
HERE = Path(__file__).resolve().parent
CHART_JSON = HERE / "site" / "components" / "charts" / "chessrl-next-round.json"
README_IMG = HERE.parent / "img"


def jl(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        return []


def run_dir(box: str, suffix: str) -> Path:
    return sorted((RUNS / box).glob(f"*_{suffix}"))[-1]


def active_hours_fn(d: Path):
    """(t0, active_hours(t)) for a run directory: hours of training, not wall clock (same rule as
    the dashboard). Self-play logs every 30 s, so a longer silence (a stop/resume, a crash) is
    downtime."""
    tr = jl(d / "metrics/train.jsonl")
    sp = jl(d / "metrics/selfplay.jsonl")
    t0 = min(r["t"] for r in tr + sp)
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

    return t0, active_hours


def load(box: str, suffix: str) -> dict:
    """The ladder as it ran (every ~30 min, 100 sims): Elo and puzzle accuracy on hours of training."""
    d = run_dir(box, suffix)
    t0, active_hours = active_hours_fn(d)
    lad = [r for r in jl(d / "ladder/ladder.jsonl") if r["t"] >= t0 - 1800]
    return {
        "hours": [active_hours(r["t"]) for r in lad],
        "elo": [r["elo"] for r in lad],
        "puzzles": [(r.get("puzzles") or {}).get("line") for r in lad],
    }


def export_json() -> None:
    """The data for the blog's native SVG charts (docs/blog/site/components/charts/).

    The two 24 h baselines run side by side on 2026-10-27, both rated live on the balanced-opening
    ladder (every ~30 min, 100 sims): the first post's code and the new recipe. Hours are training
    hours (the move to new machines after ~5.5 h is not counted)."""

    def pts(s: dict, y: str = "elo") -> list:
        return [[round(x, 2), round(v, 3 if y == "puzzles" else None)]
                for x, v in zip(s["hours"], s[y], strict=True) if v is not None]

    runs = [("Original code (first post)", 4, load("main-q", "base-main")),
            ("New recipe", 1, load("wd-q", "base-wd"))]
    data = {
        "speedup": [{"name": n, "slot": k, "points": pts(s)} for n, k, s in runs],
        "puzzles": [{"name": n, "slot": k, "points": pts(s, y="puzzles")} for n, k, s in runs],
        # Largest |x| entering the final RMSNorm, measured in FP32 PyTorch on 512 puzzle positions.
        "activation": [
            {"name": "no weight decay", "slot": 2,
             "points": [[10, 46.0], [20, 101.3], [30, 194.0], [40, 280.2], [45.5, 349.3]]},
            {"name": "weight decay 0.05", "slot": 7,
             "points": [[10, 24.9], [20, 35.7], [30, 45.8], [50, 64.4], [70, 79.9]]},
        ],
    }
    CHART_JSON.parent.mkdir(parents=True, exist_ok=True)
    CHART_JSON.write_text(json.dumps(data, indent=1) + "\n")
    print("wrote", CHART_JSON)


# GitHub's page colours, and the post's series colours for each theme (charts.css, slots 1 and 4).
THEMES = {
    "light": {"ink": "#1f2328", "muted": "#59636e", "grid": "#eaeef2", "axis": "#d1d9e0",
              "surface": "#ffffff", "series": {1: "#2a78d6", 4: "#eda100"}},
    "dark": {"ink": "#e6edf3", "muted": "#9198a1", "grid": "#21262d", "axis": "#3d444d",
             "surface": "#0d1117", "series": {1: "#3987e5", 4: "#c98500"}},
}
README_CHARTS = [  # (file name, data key, title, y label, value format, y range)
    ("elo", "speedup", "Rating by hours of training, one RTX 4090 each", "ladder Elo (random play = 0)",
     lambda v: f"{v:,.0f}", (0, 2400)),
    ("puzzles", "puzzles", "Lichess puzzles solved by hours of training", "puzzles solved (of 1,000)",
     lambda v: f"{v:.0%}", (0, 0.8)),
]


def readme_charts() -> None:
    """docs/img/{elo,puzzles}_{light,dark}.png, from the exported post data."""
    data = json.loads(CHART_JSON.read_text())
    README_IMG.mkdir(parents=True, exist_ok=True)
    for theme, c in THEMES.items():
        for name, key, title, ylabel, fmt, ylim in README_CHARTS:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(c["axis"])
            ax.grid(True, axis="y", color=c["grid"], linewidth=1)
            ax.set_axisbelow(True)
            ax.tick_params(colors=c["axis"], labelcolor=c["muted"], labelsize=10)
            for s in reversed(data[key]):  # the new recipe first in the legend
                xs, ys = zip(*s["points"], strict=True)
                color = c["series"][s["slot"]]
                ax.plot(xs, ys, color=color, lw=2, label=s["name"], solid_capstyle="round", solid_joinstyle="round")
                ax.plot([xs[-1]], [ys[-1]], "o", ms=7, color=color, mec=c["surface"], mew=2, zorder=5)
                ax.annotate(fmt(ys[-1]), (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                            va="center", color=c["ink"], fontsize=10, fontweight="bold")
                if key == "speedup":  # where each run reaches 1,000 Elo, between its two ratings around it
                    (xa, ya), (xb, yb) = next((p, q) for p, q in zip(s["points"], s["points"][1:]) if p[1] < 1000 <= q[1])
                    x1 = xa + (1000 - ya) / (yb - ya) * (xb - xa)
                    ax.plot([x1], [1000], "o", ms=7, mfc=c["surface"], mec=color, mew=2, zorder=5)
                    ax.annotate(f"{x1:.1f} h", (x1, 1000), xytext=(8, -13), textcoords="offset points",
                                color=c["ink"], fontsize=9.5)
            if key == "speedup":
                ax.axhline(1000, color=c["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
                ax.annotate("1,000 Elo", (3.2, 1000), xytext=(0, 5), textcoords="offset points", color=c["muted"],
                            fontsize=9.5)
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
            else:
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.set_xlim(0, 24.5)  # the last ratings land just past 24 h
            ax.set_xticks(range(0, 25, 4))
            ax.set_ylim(*ylim)
            ax.set_xlabel("hours of training", color=c["muted"], fontsize=10)
            ax.set_ylabel(ylabel, color=c["muted"], fontsize=10)
            ax.set_title(title, loc="left", color=c["ink"], fontsize=12.5, fontweight="bold", pad=26)
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False, fontsize=10,
                      labelcolor=c["ink"], borderaxespad=0.1, handlelength=1.6)
            fig.tight_layout()
            out = README_IMG / f"{name}_{theme}.png"
            fig.savefig(out, dpi=160, transparent=True, bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)
            print("wrote", out)


if __name__ == "__main__":
    if "--readme" in sys.argv:
        readme_charts()
    else:
        export_json()
