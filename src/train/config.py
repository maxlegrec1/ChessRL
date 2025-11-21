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
    dir: Path = Path("checkpoints")
    every_steps: int = 1000
    keep_last: int = 5
    build_script: Path | None = Path("build_trt.sh")
    engine_dir: Path = Path("engines")
    iteration_file: Path = Path("src/self_play/.iteration")


@dataclass
class DataConfig:
    dataset_path: Path = Path("src/self_play/selfplay_buffer.parquet")
    reload_seconds: float = 60.0


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
    amp_dtype: str = "float16"
    resume_path: Optional[Path] = None
    run_name: Optional[str] = None

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


def load_config(path: Path) -> TrainingConfig:
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
    cfg.data.dataset_path = Path(cfg.data.dataset_path)
    if cfg.resume_path:
        cfg.resume_path = Path(cfg.resume_path)
    return cfg

