"""Single-command launcher for the self-play + training loop.

Spawns:
  - self-play  (`python -m src.self_play`)  — generates training data
  - trainer    (`python -m src.train`)      — consumes the buffer

Each child's stdout/stderr is appended to artifacts/runs/{selfplay,training}.log.
The launcher forwards SIGINT/SIGTERM to both children; if either exits, the
other is terminated and the launcher exits with a non-zero status.

The trainer waits for `data.min_samples_to_start` records (see config.yaml)
before entering its training loop, so it is safe to start both processes
simultaneously on a fresh buffer.

Usage:
    uv run python -m src.run                            # default config
    uv run python -m src.run --config path/to/cfg.yaml  # custom config
    uv run python -m src.run --no-train                 # self-play only
    uv run python -m src.run --no-selfplay              # trainer only (resume an existing buffer)

All self-play hyperparameters (boards, threads, simulations, cpuct, temperature,
dirichlet, ...) live in the `selfplay:` section of the YAML config.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "artifacts" / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/train/config.yaml"),
                   help="Shared YAML config (selfplay + trainer sections).")
    p.add_argument("--no-train", action="store_true", help="Run self-play only.")
    p.add_argument("--no-selfplay", action="store_true", help="Run trainer only.")
    p.add_argument("--shutdown-grace", type=float, default=15.0,
                   help="Seconds to wait for children to exit before SIGKILL.")
    return p.parse_args()


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", buffering=1)
    fh.write(f"\n========== launched {datetime.now().isoformat(timespec='seconds')} ==========\n")
    return fh


def _spawn(cmd: List[str], log_path: Path) -> subprocess.Popen:
    log_fh = _open_log(log_path)
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group → can SIGTERM the whole tree
    )


def _terminate_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def main() -> int:
    args = parse_args()
    if args.no_selfplay and args.no_train:
        print("Nothing to launch (both --no-selfplay and --no-train).", file=sys.stderr)
        return 2

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    selfplay_log = RUNS_DIR / "selfplay.log"
    trainer_log = RUNS_DIR / "training.log"

    children: List[Tuple[str, subprocess.Popen, Path]] = []
    if not args.no_selfplay:
        cmd_sp = [sys.executable, "-m", "src.self_play", "--config", str(args.config)]
        children.append(("selfplay", _spawn(cmd_sp, selfplay_log), selfplay_log))

    if not args.no_train:
        cmd_tr = [sys.executable, "-m", "src.train", "--config", str(args.config)]
        children.append(("training", _spawn(cmd_tr, trainer_log), trainer_log))

    print(f"Launched {len(children)} subprocess(es):")
    for name, proc, log in children:
        print(f"  - {name:<8} pid={proc.pid}  log={log}")
    print("\nTail logs with:")
    for name, _, log in children:
        print(f"  tail -f {log}")
    print("\nCtrl-C to stop both processes.\n", flush=True)

    stop_requested = False

    def _request_shutdown(reason: str) -> None:
        nonlocal stop_requested
        if stop_requested:
            return
        stop_requested = True
        print(f"\n[launcher] {reason}; terminating children...", flush=True)
        for _, proc, _ in children:
            if proc.poll() is None:
                _terminate_group(proc, signal.SIGTERM)

    def _on_signal(signum, _frame):
        _request_shutdown(f"received signal {signum}")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    exit_code = 0
    try:
        while not stop_requested:
            for name, proc, _ in children:
                rc = proc.poll()
                if rc is not None:
                    print(f"[launcher] {name} exited with rc={rc}", flush=True)
                    if rc != 0:
                        exit_code = rc or 1
                    _request_shutdown(f"{name} exited")
                    break
            if not stop_requested:
                time.sleep(2)

        deadline = time.monotonic() + args.shutdown_grace
        for name, proc, _ in children:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[launcher] {name} did not exit cleanly; SIGKILL", flush=True)
                _terminate_group(proc, signal.SIGKILL)
                proc.wait()

    finally:
        for name, proc, _ in children:
            print(f"[launcher] {name} final rc={proc.poll()}", flush=True)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
