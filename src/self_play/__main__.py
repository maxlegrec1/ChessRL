from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from .engine_watch import EngineWatcher
from .metrics import MetricsTracker


SIMULATIONS = 800


DEFAULT_BUFFER = Path("artifacts/selfplay_buffer.bin")
DEFAULT_ITERATION = Path("artifacts/engines/.iteration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play data generation loop.")
    parser.add_argument("--boards", type=int, default=128, help="Number of parallel boards.")
    parser.add_argument("--buffer-path", type=Path, default=DEFAULT_BUFFER, help="Path to parquet buffer.")
    parser.add_argument("--max-buffer-size", type=int, default=10_000_000, help="Maximum buffer size.")
    parser.add_argument("--iteration-file", type=Path, default=DEFAULT_ITERATION, help="Path to .iteration CSV.")
    parser.add_argument("--poll-seconds", type=int, default=300, help="Seconds between iteration file checks.")
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=1,
        help="Number of buffer batches to accumulate before flushing to disk.",
    )
    parser.add_argument("--threads", type=int, default=1, help="Number of threads.")
    parser.add_argument("--hash-size", type=int, default=128, help="Hash size in MB.")
    parser.add_argument("--save-pgn", action="store_true", help="Save games as PGN")
    parser.add_argument("--pgn-dir", type=str, default="pgns", help="Directory to save PGNs")
    parser.add_argument("--mcts-value", action="store_true", help="Use MCTS value instead of game outcome for WDL")
    return parser.parse_args()


def render_dashboard_cpp(metrics: MetricsTracker, worker_metrics: dict, board_count: int) -> Table:
    snapshot = metrics.snapshot()
    table = Table(title="Self-Play Generator (C++)", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Metric", justify="right")
    table.add_column("Value", justify="left")

    table.add_row("Positions/hour", f"{snapshot.positions_per_hour:,.0f}")
    table.add_row("Positions/sec", f"{snapshot.positions_per_sec:,.2f}")
    table.add_row("Nodes/sec", f"{worker_metrics.get('nodes_per_sec', 0):,.0f}")
    table.add_row("Total positions", f"{snapshot.total_positions:,}")
    table.add_row("Total games", f"{snapshot.total_games:,}")
    table.add_row("Boards", f"{board_count}")
    table.add_row("Buffer size", f"{worker_metrics.get('buffer_size', 0):,}")
    table.add_row("Engine", str(snapshot.engine_path) if snapshot.engine_path else "unknown")

    if snapshot.last_engine_swap:
        age_minutes = max(0.0, (time.time() - snapshot.last_engine_swap) / 60.0)
        table.add_row("Last engine swap", f"{age_minutes:.1f} min ago")
    else:
        table.add_row("Last engine swap", "n/a")

    return table

import _inference_cpp

def main() -> int:
    args = parse_args()
    console = Console()

    buffer_path = args.buffer_path.resolve()
    iteration_path = args.iteration_file.resolve()

    # Ensure directory exists
    buffer_path.parent.mkdir(parents=True, exist_ok=True)

    engine_watcher = EngineWatcher(iteration_path, poll_interval=float(args.poll_seconds))
    metrics = MetricsTracker(simulations_per_position=SIMULATIONS)
    
    console.log(f"Starting C++ self-play loop...")
    console.log(f"Buffer: {buffer_path}")
    if args.save_pgn:
        console.log(f"Saving PGNs to: {args.pgn_dir}")
    
    worker = None
    current_engine_path = None
    
    try:
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                # Check for engine update
                new_engine = engine_watcher.maybe_refresh()
                if new_engine:
                    metrics.update_engine_path(new_engine)
                    current_engine_path = str(new_engine)
                    # Re-initialize worker with new engine
                    worker = _inference_cpp.SelfPlayWorker(
                        engine_path=current_engine_path,
                        output_path=str(buffer_path),
                        batch_size=args.boards,
                        threads=args.threads,
                        hash_size=args.hash_size,
                        max_games=-1,
                        max_buffer_size=args.max_buffer_size,
                        save_pgn=args.save_pgn,
                        pgn_dir=args.pgn_dir,
                        simulations=SIMULATIONS,
                        mcts_value=args.mcts_value
                    )
                    console.log(f"Initialized worker with engine: {current_engine_path}")

                if not worker:
                    if not current_engine_path:
                        current_engine_path = engine_watcher.engine_path
                    
                    if current_engine_path:
                        metrics.update_engine_path(Path(current_engine_path))
                        worker = _inference_cpp.SelfPlayWorker(
                            engine_path=str(current_engine_path),
                            output_path=str(buffer_path),
                            batch_size=args.boards,
                            threads=args.threads,
                            hash_size=args.hash_size,
                            max_games=-1,
                            max_buffer_size=args.max_buffer_size,
                            save_pgn=args.save_pgn,
                            pgn_dir=args.pgn_dir,
                            simulations=SIMULATIONS,
                            mcts_value=args.mcts_value
                        )
                        console.log(f"Initialized worker with engine: {current_engine_path}")
                    else:
                        live.update(Table(title="Waiting for engine...", box=box.SIMPLE_HEAVY))
                        time.sleep(5)
                        continue
                
                # Step the worker
                step_metrics = worker.step()
                
                # Update metrics
                metrics.record_positions(step_metrics.positions_generated)
                metrics.record_game_finished(step_metrics.games_finished)
                
                # Calculate nodes per second for this step
                nps = 0
                if step_metrics.runtime_seconds > 0:
                    nps = step_metrics.nodes_evaluated / step_metrics.runtime_seconds
                
                # Get buffer size
                buffer_size = worker.get_buffer_size()
                
                worker_stats = {
                    "nodes_per_sec": nps,
                    "buffer_size": buffer_size
                }
                
                live.update(render_dashboard_cpp(metrics, worker_stats, args.boards))
                
    except KeyboardInterrupt:
        console.log("Stopping self-play loop...")
        return 0
    except Exception:
        console.print_exception()
        return 1

if __name__ == "__main__":
    sys.exit(main())

