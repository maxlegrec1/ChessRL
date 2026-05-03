from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class WandbConfig:
    project: str = "self_play"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    enabled: bool = True


@dataclass
class CheckpointConfig:
    dir: Path = Path("artifacts/checkpoints")
    every_steps: int = 1000
    keep_last: int = 5
    build_script: Path | None = Path("scripts/build_trt.sh")
    engine_dir: Path = Path("artifacts/engines")
    iteration_file: Path = Path("artifacts/engines/.iteration")


@dataclass
class DataConfig:
    dataset_path: Path = Path("artifacts/buffers/buffer.bin")
    reload_seconds: float = 60.0
    min_samples_to_start: int = 8192
    warmup_poll_seconds: float = 10.0


@dataclass
class SelfPlayConfig:
    # parallelism / batching
    boards: int = 128
    threads: int = 1
    hash_size: int = 128
    # MCTS hyperparameters (mirror cpp/selfplay.hpp defaults)
    simulations: int = 800
    cpuct: float = 0.65
    temperature: float = 1.3
    temperature_at_root: float = 1.0
    use_dirichlet: bool = True
    dirichlet_alpha: float = 0.12
    dirichlet_epsilon: float = 0.05
    # buffer / output
    max_buffer_size: int = 10_000_000
    flush_interval: int = 1
    poll_seconds: int = 300
    save_pgn: bool = False
    pgn_dir: Path = Path("artifacts/pgns")
    mcts_value: bool = False


@dataclass
class TrainingConfig:
    device: str = "cuda"
    batch_size: int = 256
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    log_every: int = 50
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    amp_dtype: str = "float16"
    resume_path: Optional[Path] = None
    run_name: Optional[str] = None
    teacher_decay: float = 0.8
    teacher_weight: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance, updates: Dict[str, Any]):
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        attr = getattr(instance, key)
        if isinstance(attr, Path):
            setattr(instance, key, Path(value) if value is not None else None)
        elif hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(attr, value)
        else:
            setattr(instance, key, value)


def load_config(path: Path | str) -> TrainingConfig:
    path = Path(path)
    cfg = TrainingConfig()
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge_dataclass(cfg, data)
    cfg.checkpoint.dir = Path(cfg.checkpoint.dir)
    if cfg.checkpoint.build_script is not None:
        cfg.checkpoint.build_script = Path(cfg.checkpoint.build_script)
    cfg.checkpoint.engine_dir = Path(cfg.checkpoint.engine_dir)
    cfg.checkpoint.iteration_file = Path(cfg.checkpoint.iteration_file)
    cfg.selfplay.pgn_dir = Path(cfg.selfplay.pgn_dir)
    cfg.data.dataset_path = Path(cfg.data.dataset_path)
    if cfg.resume_path:
        cfg.resume_path = Path(cfg.resume_path)
    return cfg

