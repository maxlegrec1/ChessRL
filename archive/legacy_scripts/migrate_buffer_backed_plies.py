"""Extend a legacy 144-byte self-play buffer to the new 148-byte layout.

The MCTS-backed plies-remaining target (mlh_q) was added after the original
144-byte record format shipped. New records are 148 bytes; the new field at
offset 144 holds `backed_plies` (raw plies estimate from MCTS root). Old rows
get -1.0 (the "no target" sentinel the trainer's mlh_q mask filters out).

This script also stamps the existing plies_remaining slot with -1 wherever
it currently reads 0.0 — see migrate_buffer_mlh_sentinel.py for context. So a
buffer migrated through this script is consumable by the new trainer with no
mistraining on legacy "every position is mate-now" rows.

Usage:
    uv run python scripts/migrate_buffer_backed_plies.py <buffer.bin>

Idempotent: reading a 148-byte file is detected and a no-op pass over the
plies_remaining/backed_plies slots is performed (no rewrites).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

OLD_RECORD = 144
NEW_RECORD = 148
PLIES_OFFSET = 140
BACKED_PLIES_OFFSET = 144
SENTINEL_BYTES = struct.pack("f", -1.0)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1

    size = path.stat().st_size

    if size % NEW_RECORD == 0 and size % OLD_RECORD != 0:
        # Already migrated; nothing to do beyond the legacy plies sentinel pass.
        n = size // NEW_RECORD
        print(f"buffer already in new 148-byte format: {n:,} records — checking sentinels")
        with path.open("r+b") as f:
            stamped = 0
            for i in range(n):
                f.seek(i * NEW_RECORD + PLIES_OFFSET)
                (cur,) = struct.unpack("f", f.read(4))
                if cur == 0.0:
                    f.seek(i * NEW_RECORD + PLIES_OFFSET)
                    f.write(SENTINEL_BYTES)
                    stamped += 1
            f.flush()
        print(f"  stamped {stamped:,} pre-MLH plies_remaining slots to -1")
        return 0

    if size % OLD_RECORD != 0:
        print(
            f"file size {size} is not a multiple of {OLD_RECORD} (legacy) "
            f"or {NEW_RECORD} (new); aborting to avoid corrupting data",
            file=sys.stderr,
        )
        return 1

    n = size // OLD_RECORD
    print(f"migrating {n:,} legacy 144-byte records → 148-byte layout in {path}")

    # Stream rewrite: read each 144-byte record, append 4 sentinel bytes,
    # write to a sibling tmp file, then rename atomically. Avoids leaving a
    # half-migrated buffer if interrupted.
    tmp = path.with_suffix(path.suffix + ".migrating")
    with path.open("rb") as src, tmp.open("wb") as dst:
        for i in range(n):
            rec = src.read(OLD_RECORD)
            if len(rec) != OLD_RECORD:
                print(f"unexpected EOF at record {i}; aborting", file=sys.stderr)
                tmp.unlink(missing_ok=True)
                return 1
            (plies,) = struct.unpack_from("f", rec, PLIES_OFFSET)
            if plies == 0.0:
                # Pre-MLH zero-padded slot; rewrite to -1 sentinel in-place.
                rec = rec[:PLIES_OFFSET] + SENTINEL_BYTES + rec[PLIES_OFFSET + 4:]
            dst.write(rec)
            dst.write(SENTINEL_BYTES)  # backed_plies = -1
    tmp.replace(path)
    print(f"  rewrote {n:,} records; backed_plies set to -1 sentinel for all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
