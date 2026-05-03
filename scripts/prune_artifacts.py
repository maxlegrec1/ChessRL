"""Prune artifacts/checkpoints and artifacts/engines to the last N step numbers.

Usage:
    uv run python scripts/prune_artifacts.py                    # dry-run
    uv run python scripts/prune_artifacts.py --apply
    uv run python scripts/prune_artifacts.py --keep-last 5 --apply
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

CKPT_DIR = Path("artifacts/checkpoints")
ENG_DIR = Path("artifacts/engines")

STEP_PT_RE = re.compile(r"^step_(\d+)\.pt$")
ANY_STEP_RE = re.compile(r"_(\d+)\.(?:pt|onnx|trt)$")

ALWAYS_KEEP = {"initial_teacher.pt", "model_dynamic_initial.trt"}


def collect_step_numbers(directory: Path) -> list[int]:
    if not directory.exists():
        return []
    steps = set()
    for p in directory.iterdir():
        m = STEP_PT_RE.match(p.name)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-last", type=int, default=5)
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete (default: dry-run).")
    args = parser.parse_args()

    steps = collect_step_numbers(CKPT_DIR)
    if not steps:
        print(f"No step_*.pt found in {CKPT_DIR}; nothing to do.")
        return 0

    keep = set(steps[-args.keep_last:])
    drop = sorted(set(steps) - keep)

    print(f"Found {len(steps)} step checkpoints in {CKPT_DIR}")
    print(f"Keeping last {len(keep)}: {sorted(keep)}")
    print(f"Dropping {len(drop)} step numbers")
    if not args.apply:
        print("(dry-run; pass --apply to delete)")

    deleted = 0
    freed = 0
    for directory in (CKPT_DIR, ENG_DIR):
        if not directory.exists():
            continue
        for p in sorted(directory.iterdir()):
            if p.name in ALWAYS_KEEP:
                continue
            m = ANY_STEP_RE.search(p.name)
            if not m:
                continue
            step = int(m.group(1))
            if step in keep:
                continue
            size = p.stat().st_size if p.is_file() else 0
            print(("DELETE " if args.apply else "[dry] ") + str(p))
            if args.apply:
                p.unlink()
            deleted += 1
            freed += size

    print(f"{'Deleted' if args.apply else 'Would delete'} {deleted} files, "
          f"freeing ~{freed / 1e9:.2f} GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
