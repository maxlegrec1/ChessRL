from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from .buffer import SelfPlayBuffer
from .engine_watch import EngineWatcher
from .loop import SIMULATIONS, SelfPlayLoop
from .metrics import MetricsTracker


DEFAULT_BUFFER = Path("src/self_play/selfplay_buffer.parquet")
DEFAULT_ITERATION = Path("src/self_play/.iteration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play data generation loop.")
    parser.add_argument("--boards", type=int, default=128, help="Number of parallel boards.")
    parser.add_argument("--buffer-path", type=Path, default=DEFAULT_BUFFER, help="Path to parquet buffer.")
    parser.add_argument("--iteration-file", type=Path, default=DEFAULT_ITERATION, help="Path to .iteration CSV.")
    parser.add_argument("--poll-seconds", type=int, default=600, help="Seconds between iteration file checks.")
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=1,
        help="Number of buffer batches to accumulate before flushing to disk.",
    )
    return parser.parse_args()


def render_dashboard(loop: SelfPlayLoop, metrics: MetricsTracker) -> Table:
    snapshot = metrics.snapshot()
    table = Table(title="Self-Play Generator", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")

    table.add_row("Positions/hour", f"{snapshot.positions_per_hour:,.0f}")
    table.add_row("Positions/sec", f"{snapshot.positions_per_sec:,.2f}")
    table.add_row("Nodes/sec", f"{snapshot.nodes_per_sec:,.0f}")
    table.add_row("Total positions", f"{snapshot.total_positions:,}")
    table.add_row("Total games", f"{snapshot.total_games:,}")
    table.add_row("Boards", f"{loop.board_count}")
    table.add_row("Buffer fill", f"{snapshot.buffer_fill_pct:5.1f}%")
    table.add_row("Engine", snapshot.engine_path or "unknown")

    if snapshot.last_engine_swap:
        age_minutes = max(0.0, (time.time() - snapshot.last_engine_swap) / 60.0)
        table.add_row("Last engine swap", f"{age_minutes:.1f} min ago")
    else:
        table.add_row("Last engine swap", "n/a")

    return table


def main() -> int:
    args = parse_args()
    console = Console()

    buffer_path = args.buffer_path.resolve()
    iteration_path = args.iteration_file.resolve()

    buffer = SelfPlayBuffer(buffer_path, capacity=1_000_000, flush_interval=args.flush_interval)
    existing = buffer.load()

    engine_watcher = EngineWatcher(iteration_path, poll_interval=float(args.poll_seconds))
    metrics = MetricsTracker(simulations_per_position=SIMULATIONS)
    metrics.update_buffer_fill(buffer.fill_ratio())
    metrics.update_engine_path(engine_watcher.engine_path)

    console.log(f"Loaded buffer rows: {existing}")
    console.log(f"Current engine: {engine_watcher.engine_path}")

    loop = SelfPlayLoop(
        board_count=args.boards,
        buffer=buffer,
        metrics=metrics,
        engine_watcher=engine_watcher,
    )

    try:
        with Live(render_dashboard(loop, metrics), refresh_per_second=1, console=console) as live:
            while True:
                loop.step()
                live.update(render_dashboard(loop, metrics))
    except KeyboardInterrupt:
        console.log("Stopping self-play loop...")
        return 0
    except Exception:
        console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())

