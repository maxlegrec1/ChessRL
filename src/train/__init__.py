"""Self-play training package."""

from .config import TrainingConfig, load_config
from .trainer import Trainer

__all__ = ["TrainingConfig", "load_config", "Trainer"]
