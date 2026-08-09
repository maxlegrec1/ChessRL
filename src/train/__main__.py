from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BT4 on self-play data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train/config.yaml"),
        help="Path to training YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
