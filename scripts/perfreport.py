"""Render a side-by-side markdown summary of two bench_selfplay.json files.

Usage:
  uv run python scripts/perfreport.py artifacts/perf/phase0.json artifacts/perf/phase1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def fmt(v: float, places: int = 2) -> str:
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.{places}f}"
    return f"{v:.{places}f}"


def delta_pct(a: float, b: float) -> str:
    if a == 0 or a is None or b is None:
        return "-"
    return f"{(b / a - 1) * 100:+.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline JSON")
    ap.add_argument("b", help="comparison JSON")
    ap.add_argument("--out", default=None, help="write markdown to a file as well as stdout")
    args = ap.parse_args()

    A = json.loads(pathlib.Path(args.a).read_text())
    B = json.loads(pathlib.Path(args.b).read_text())

    rows = [
        (
            "positions/sec",
            A["throughput"]["positions_per_sec"],
            B["throughput"]["positions_per_sec"],
        ),
        ("games/sec", A["throughput"]["games_per_sec"], B["throughput"]["games_per_sec"]),
        ("nodes/sec", A["throughput"]["nodes_per_sec"], B["throughput"]["nodes_per_sec"]),
        ("mean batch size", A["inference"]["mean_batch_size"], B["inference"]["mean_batch_size"]),
        (
            "ms / batch",
            A["inference"]["mean_infer_ms_per_batch"],
            B["inference"]["mean_infer_ms_per_batch"],
        ),
        ("us / encoder leaf", A["encoder"]["mean_us_per_leaf"], B["encoder"]["mean_us_per_leaf"]),
        (
            "encoder thread busy %",
            A["encoder"]["thread_busy_frac"] * 100,
            B["encoder"]["thread_busy_frac"] * 100,
        ),
    ]

    md: list[str] = []
    md.append(f"# Self-play perf: `{A['label']}` vs `{B['label']}`")
    md.append("")
    md.append(f"- GPU: {A.get('gpu', '?')} vs {B.get('gpu', '?')}")
    md.append(f"- Engine A: `{A['engine']}`  ({A['steps']} steps in {A['wall_clock_sec']:.1f}s)")
    md.append(f"- Engine B: `{B['engine']}`  ({B['steps']} steps in {B['wall_clock_sec']:.1f}s)")
    md.append(
        f"- Config: concurrency={A['config']['concurrency']}/{B['config']['concurrency']}  "
        f"threads={A['config']['threads']}/{B['config']['threads']}  "
        f"sims={A['config']['sims']}/{B['config']['sims']}"
    )
    md.append("")
    md.append(f"| metric | {A['label']} | {B['label']} | Δ |")
    md.append("|---|---:|---:|---:|")
    for label, a, b in rows:
        md.append(f"| {label} | {fmt(a)} | {fmt(b)} | {delta_pct(a, b)} |")

    text = "\n".join(md) + "\n"
    print(text)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
