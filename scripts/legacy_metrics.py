"""Metrics adapter for a run produced by the original main-branch code.

main logs only to W&B and Rich consoles, so this follows such a run and writes
the same JSONL files the next-round code produces (enough for the dashboard
and the blog charts):
  metrics/selfplay.jsonl  t, total_records_written (from buffer.bin.writes), positions_per_sec, engine_step
  metrics/train.jsonl     step, loss/total, acc/policy, acc/value (parsed from training.log)

  uv run python scripts/legacy_metrics.py --run-dir /workspace/ChessRL-main/artifacts/runs/<id>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

STEP_RE = re.compile(
    r"\[step\s+(\d+)\]\s+loss=([\d.]+)\s+policy_acc=([\d.]+)\s+value_acc=([\d.]+)", re.S
)


def engine_step(run_dir: Path) -> int:
    it = run_dir / "engines" / ".iteration"
    try:
        rows = list(csv.DictReader(it.open(), skipinitialspace=True))
        return int(rows[-1]["iteration"]) if rows else 0
    except (FileNotFoundError, ValueError, KeyError):
        return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--every", type=float, default=30.0)
    a = p.parse_args()
    metrics = a.run_dir / "metrics"
    metrics.mkdir(exist_ok=True)
    writes_file = a.run_dir / "buffer" / "buffer.bin.writes"
    log = a.run_dir / "training.log"
    last_writes, last_t, offset, last_step = None, None, 0, -1
    while True:
        now = time.time()
        try:
            writes = int(writes_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            writes = 0
        row = {"t": now, "total_records_written": writes, "engine_step": engine_step(a.run_dir)}
        if last_writes is not None:
            row["positions_per_sec"] = (writes - last_writes) / max(1e-6, now - last_t)
        with (metrics / "selfplay.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        last_writes, last_t = writes, now

        if log.exists():
            with log.open("r", errors="replace") as fh:
                fh.seek(offset)
                text = fh.read()
            # Keep a tail in case a log line was cut mid-write.
            cut = max(0, len(text) - 2000)
            offset += cut
            rows = []
            for m in STEP_RE.finditer(text):
                step = int(m.group(1))
                if step > last_step:
                    rows.append({"step": step, "t": now, "loss/total": float(m.group(2)),
                                 "acc/policy": float(m.group(3)), "acc/value": float(m.group(4)),
                                 "buffer/total_writes": writes})
                    last_step = step
            if rows:
                with (metrics / "train.jsonl").open("a") as fh:
                    fh.writelines(json.dumps(r) + "\n" for r in rows)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
