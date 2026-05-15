from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import _inference_cpp
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.train.config import SelfPlayConfig, TrainingConfig, load_config

from .engine_watch import EngineWatcher
from .metrics import MetricsTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play data generation loop.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/train/config.yaml"),
        help="YAML config (selfplay + checkpoint + data sections).",
    )
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


def _build_worker(engine_path: str, buffer_path: Path, sp: SelfPlayConfig):
    return _inference_cpp.SelfPlayWorker(
        engine_path=engine_path,
        output_path=str(buffer_path),
        batch_size=sp.boards,
        threads=sp.threads,
        hash_size=sp.hash_size,
        max_games=-1,
        max_buffer_size=sp.max_buffer_size,
        save_pgn=sp.save_pgn,
        pgn_dir=str(sp.pgn_dir),
        simulations=sp.simulations,
        cpuct=sp.cpuct,
        temperature=sp.temperature,
        temperature_at_root=sp.temperature_at_root,
        use_dirichlet=sp.use_dirichlet,
        dirichlet_alpha=sp.dirichlet_alpha,
        dirichlet_epsilon=sp.dirichlet_epsilon,
        random_opening_plies=sp.random_opening_plies,
        virtual_loss_width=sp.virtual_loss_width,
    )


def main() -> int:
    args = parse_args()
    console = Console()

    config: TrainingConfig = load_config(args.config)
    sp = config.selfplay

    buffer_path = config.data.dataset_path.resolve()
    iteration_path = config.checkpoint.iteration_file.resolve()
    buffer_path.parent.mkdir(parents=True, exist_ok=True)

    engine_watcher = EngineWatcher(iteration_path, poll_interval=float(sp.poll_seconds))
    metrics = MetricsTracker(simulations_per_position=sp.simulations)

    console.log("Starting C++ self-play loop...")
    console.log(f"Buffer: {buffer_path}")
    console.log(
        f"MCTS: simulations={sp.simulations} cpuct={sp.cpuct} "
        f"temperature={sp.temperature} root_temp={sp.temperature_at_root} "
        f"dirichlet={sp.use_dirichlet}(alpha={sp.dirichlet_alpha}, eps={sp.dirichlet_epsilon}) "
        f"random_opening_plies={sp.random_opening_plies}"
    )
    if sp.save_pgn:
        console.log(f"Saving PGNs to: {sp.pgn_dir}")

    worker = None
    current_engine_path: str | None = None

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                new_engine = engine_watcher.maybe_refresh()
                if new_engine:
                    metrics.update_engine_path(new_engine)
                    current_engine_path = str(new_engine)
                    worker = _build_worker(current_engine_path, buffer_path, sp)
                    console.log(f"Initialized worker with engine: {current_engine_path}")

                if not worker:
                    if not current_engine_path:
                        current_engine_path = engine_watcher.engine_path
                    if current_engine_path:
                        metrics.update_engine_path(Path(current_engine_path))
                        worker = _build_worker(str(current_engine_path), buffer_path, sp)
                        console.log(f"Initialized worker with engine: {current_engine_path}")
                    else:
                        live.update(Table(title="Waiting for engine...", box=box.SIMPLE_HEAVY))
                        time.sleep(5)
                        continue

                step_metrics = worker.step()
                metrics.record_positions(step_metrics.positions_generated)
                metrics.record_game_finished(step_metrics.games_finished)

                nps = 0
                if step_metrics.runtime_seconds > 0:
                    nps = step_metrics.nodes_evaluated / step_metrics.runtime_seconds

                buffer_size = worker.get_buffer_size()
                worker_stats = {"nodes_per_sec": nps, "buffer_size": buffer_size}

                live.update(render_dashboard_cpp(metrics, worker_stats, sp.boards))

    except KeyboardInterrupt:
        console.log("Stopping self-play loop...")
        return 0
    except Exception:
        console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
