"""Stamp every record in a self-play buffer with plies_remaining = -1.

The MLH head was added after the buffer was populated, so existing rows have
0.0 in the plies slot (zero-padded by the C++ writer's initial memset). The
trainer treats -1 as "no MLH target" so the head doesn't get mistrained on
"every position is mate-now."

After migration, only freshly written self-play records will have valid
plies_remaining values; the buffer's MLH-trainable fraction grows as
random-replacement cycles new data in.

Usage:
    uv run python scripts/migrate_buffer_mlh_sentinel.py <buffer.bin>

Idempotent: reading -1 back and writing -1 again is a no-op.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

RECORD_SIZE = 144
PLIES_OFFSET = 140
SENTINEL = -1.0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1

    size = path.stat().st_size
    if size % RECORD_SIZE != 0:
        print(
            f"warning: file size {size} is not a multiple of {RECORD_SIZE}; "
            f"trailing partial record will be ignored",
            file=sys.stderr,
        )

    n_records = size // RECORD_SIZE
    sentinel_bytes = struct.pack("f", SENTINEL)
    print(f"migrating {n_records:,} records in {path}")

    with path.open("r+b") as f:
        # Pass 1: count how many already have a valid (non-zero) target so we
        # know whether to skip work.
        non_zero = 0
        for i in range(n_records):
            f.seek(i * RECORD_SIZE + PLIES_OFFSET)
            (cur,) = struct.unpack("f", f.read(4))
            if cur != 0.0:
                non_zero += 1
        print(
            f"  pre-migration: {non_zero:,} records already have a non-zero "
            f"plies field ({100 * non_zero / max(1, n_records):.2f}%)"
        )

        # Pass 2: stamp -1 over rows that currently read 0.0 (the zero-padded
        # legacy slot). Never overwrite a row that already has a real target,
        # so a partial run is safely re-runnable.
        rewritten = 0
        for i in range(n_records):
            f.seek(i * RECORD_SIZE + PLIES_OFFSET)
            (cur,) = struct.unpack("f", f.read(4))
            if cur == 0.0:
                f.seek(i * RECORD_SIZE + PLIES_OFFSET)
                f.write(sentinel_bytes)
                rewritten += 1
        f.flush()

    print(
        f"  rewrote {rewritten:,} records to plies_remaining=-1; "
        f"left {n_records - rewritten:,} pre-existing valid targets intact"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
