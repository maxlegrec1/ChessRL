#!/usr/bin/env python3
"""Quick check for null/OOV policy moves in the self-play buffer."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.small.vocab import policy_to_idx


RECORD_SIZE = 128
FEN_BYTES = 100
POLICY_BYTES = 16


def inspect_buffer(path: Path, sample_limit: int = 5) -> None:
    if not path.exists():
        print(f"File not found: {path}")
        return

    total = 0
    null_count = 0
    samples: list[str] = []

    with path.open("rb") as f:
        while True:
            chunk = f.read(RECORD_SIZE)
            if len(chunk) < RECORD_SIZE:
                break

            policy_bytes = chunk[FEN_BYTES : FEN_BYTES + POLICY_BYTES]
            try:
                policy = policy_bytes.split(b"\0", 1)[0].decode("utf-8").strip()
            except Exception:
                policy = ""

            total += 1
            if not policy or policy not in policy_to_idx:
                null_count += 1
                if len(samples) < sample_limit:
                    samples.append(policy if policy else "<empty>")

    pct = (null_count / total * 100) if total else 0.0
    print(f"Scanned: {total:,} rows in {path}")
    print(f"Null / OOV policies: {null_count:,} ({pct:.2f}%)")
    if samples:
        print("\nSample null/OOV entries:")
        for i, p in enumerate(samples, 1):
            print(f"  {i}. {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="buffer_value_perspective.bin",
        help="Path to buffer file (default: buffer_value_perspective.bin)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="How many sample null/OOV policies to print",
    )
    args = parser.parse_args()
    inspect_buffer(Path(args.path), sample_limit=args.samples)


if __name__ == "__main__":
    main()

