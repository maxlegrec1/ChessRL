from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _inference_cpp
from rich.console import Console

from src.train.config import SelfPlayConfig, TrainingConfig, load_config

from .calibration import calibrate_resign
from .engine_watch import EngineWatcher

# Fields copied from the Python SelfPlayConfig onto the C++ one (same names).
_CPP_FIELDS = (
    "threads",
    "hash_size",
    "simulations",
    "cpuct",
    "temperature",
    "temperature_at_root",
    "virtual_loss_width",
    "use_dirichlet",
    "dirichlet_alpha",
    "dirichlet_epsilon",
    "max_buffer_size",
    "save_pgn",
    "random_opening_plies",
    "temperature_plies",
    "temperature_late",
    "resign_threshold",
    "resign_consecutive",
    "resign_playthrough",
    "max_plies",
    "pcr_full_prob",
    "pcr_fast_simulations",
    "quick_policy_temperature",
    "fifo_buffer",
    "curriculum_prob",
    "curriculum_max_pieces",
)

_STAT_FIELDS = (
    "games",
    "white_wins",
    "black_wins",
    "draws",
    "game_plies",
    "end_checkmate",
    "end_resign",
    "end_max_plies",
    "full_searches",
    "fast_searches",
    "policy_moves",
    "records_written",
    "curriculum_games",
    "playthrough_games",
    "playthrough_would_resign",
    "playthrough_false_resign",
)

METRICS_EVERY_S = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play data generation loop.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/train/config.yaml"),
        help="YAML config (selfplay + checkpoint + data sections).",
    )
    return parser.parse_args()


def curriculum_prob_at(sp: SelfPlayConfig, trainer_step: int) -> float:
    """curriculum_prob linearly annealed to 0 over curriculum_anneal_steps."""
    if sp.curriculum_anneal_steps <= 0:
        return sp.curriculum_prob
    return sp.curriculum_prob * max(0.0, 1.0 - trainer_step / sp.curriculum_anneal_steps)


def build_cpp_config(
    sp: SelfPlayConfig, engine_path: str, buffer_path: Path, trainer_step: int
) -> _inference_cpp.SelfPlayConfig:
    cfg = _inference_cpp.SelfPlayConfig()
    for name in _CPP_FIELDS:
        setattr(cfg, name, getattr(sp, name))
    cfg.engine_path = engine_path
    cfg.output_path = str(buffer_path)
    cfg.concurrency = sp.boards
    cfg.max_games = -1
    cfg.pgn_dir = str(sp.pgn_dir)
    cfg.curriculum_prob = curriculum_prob_at(sp, trainer_step)
    return cfg


def main() -> int:
    args = parse_args()
    console = Console()

    config: TrainingConfig = load_config(args.config)
    sp = config.selfplay

    buffer_path = config.data.dataset_path.resolve()
    iteration_path = config.checkpoint.iteration_file.resolve()
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_dir = config.checkpoint.dir.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "selfplay.jsonl"

    engine_watcher = EngineWatcher(iteration_path, poll_interval=float(sp.poll_seconds))

    console.log("Starting C++ self-play loop...")
    console.log(f"Buffer: {buffer_path}")
    console.log(
        f"MCTS: simulations={sp.simulations} cpuct={sp.cpuct} "
        f"temperature={sp.temperature} root_temp={sp.temperature_at_root} "
        f"temp_plies={sp.temperature_plies} late_temp={sp.temperature_late} "
        f"dirichlet={sp.use_dirichlet}(alpha={sp.dirichlet_alpha}, eps={sp.dirichlet_epsilon}) "
        f"random_opening_plies={sp.random_opening_plies} resign={sp.resign_threshold} "
        f"pcr={sp.pcr_full_prob}/{sp.pcr_fast_simulations} fifo={sp.fifo_buffer} "
        f"curriculum={sp.curriculum_prob}(anneal {sp.curriculum_anneal_steps})"
    )

    worker = None
    current_curriculum = None
    resign_mark = None  # (would, false) counters at the last calibration
    window_t = time.time()
    window_positions = 0
    window_nodes = 0
    window_games = 0
    window_steps = 0
    window_runtime = 0.0

    try:
        while True:
            new_engine = engine_watcher.maybe_refresh()
            if worker is None:
                engine_path = engine_watcher.engine_path
                if not engine_path:
                    console.log("Waiting for an engine in .iteration ...")
                    time.sleep(5)
                    engine_watcher._refresh(force=True)
                    continue
                step = engine_watcher.engine_step
                worker = _inference_cpp.SelfPlayWorker(
                    build_cpp_config(sp, engine_path, buffer_path, step)
                )
                current_curriculum = curriculum_prob_at(sp, step)
                console.log(f"Initialized worker with engine: {engine_path} (step {step})")
            elif new_engine:
                # Swap in place: games in progress keep going on the new net.
                worker.set_engine(new_engine)
                step = engine_watcher.engine_step
                curriculum = curriculum_prob_at(sp, step)
                if curriculum != current_curriculum:
                    worker.set_config(build_cpp_config(sp, new_engine, buffer_path, step))
                    current_curriculum = curriculum
                console.log(f"Swapped engine: {new_engine} (step {step}, curriculum {curriculum:.3f})")

            step_metrics = worker.step()
            window_positions += step_metrics.positions_generated
            window_nodes += step_metrics.nodes_evaluated
            window_games += step_metrics.games_finished
            window_runtime += step_metrics.runtime_seconds
            window_steps += 1

            now = time.time()
            if now - window_t >= METRICS_EVERY_S:
                dt = now - window_t
                stats = worker.stats()
                if sp.resign_target_fp > 0 and sp.resign_threshold > 0:
                    if resign_mark is None:
                        resign_mark = (stats.playthrough_would_resign, stats.playthrough_false_resign)
                    would = stats.playthrough_would_resign - resign_mark[0]
                    false = stats.playthrough_false_resign - resign_mark[1]
                    if would >= sp.resign_calibration_window:
                        new = calibrate_resign(sp.resign_threshold, would, false, sp.resign_target_fp)
                        console.log(f"[resign] window fp={false / would:.3f} threshold "
                                    f"{sp.resign_threshold:.4f} -> {new:.4f}")
                        if new != sp.resign_threshold:
                            sp.resign_threshold = new
                            worker.set_config(build_cpp_config(sp, engine_watcher.engine_path, buffer_path,
                                                               engine_watcher.engine_step))
                        resign_mark = (stats.playthrough_would_resign, stats.playthrough_false_resign)
                row = {
                    "t": now,
                    "engine_step": engine_watcher.engine_step,
                    "positions_per_sec": window_positions / dt,
                    "nodes_per_sec": window_nodes / dt,
                    "games_per_sec": window_games / dt,
                    "step_seconds": window_runtime / max(1, window_steps),
                    "buffer_records": worker.get_buffer_size(),
                    "curriculum_prob": current_curriculum,
                    "resign_threshold": sp.resign_threshold,
                    **{f"total_{k}": getattr(stats, k) for k in _STAT_FIELDS},
                }
                with metrics_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                console.log(
                    f"pos/s={row['positions_per_sec']:.1f} nodes/s={row['nodes_per_sec']:,.0f} "
                    f"games/s={row['games_per_sec']:.2f} games={stats.games:,} "
                    f"records={stats.records_written:,} engine_step={row['engine_step']}"
                )
                window_t = now
                window_positions = window_nodes = window_games = window_steps = 0
                window_runtime = 0.0

    except KeyboardInterrupt:
        console.log("Stopping self-play loop...")
        return 0
    except Exception:
        console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
