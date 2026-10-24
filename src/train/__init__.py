"""Self-play training package."""

from .config import TrainingConfig, load_config

__all__ = ["TrainingConfig", "load_config", "Trainer"]


def __getattr__(name: str):
    # Lazy so `import src.train.dataset` doesn't pull in torch-heavy trainer deps.
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(name)
