"""Single-command launcher with per-run isolation.

By default each launch creates a fresh run directory at
    artifacts/runs/<run_id>/
containing isolated subfolders for the buffer, engines, checkpoints, pgns and
wandb logs, plus a snapshot of the resolved config used for the run. A
random-initialized network (.cnet) is built up-front so self-play has something
to play with from the very first second.

Usage:
    uv run python -m src.run                          # fresh run
    uv run python -m src.run --name my-experiment     # fresh run, named
    uv run python -m src.run --resume artifacts/runs/<id>   # resume an existing run

Each child's stdout/stderr is appended to <run_dir>/{selfplay,training}.log.
SIGINT/SIGTERM is forwarded to both children; if either exits the launcher
terminates the other and exits with a non-zero status.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "artifacts" / "runs"
DEFAULT_BASE_CONFIG = ROOT / "src" / "train" / "config.yaml"
SUBDIRS = ("buffer", "engines", "checkpoints", "pgns", "wandb")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Base YAML config (paths inside it are overridden per-run). Default: "
        "src/train/config.yaml for a new run, the run's own config.yaml on --resume.",
    )
    p.add_argument(
        "--name", type=str, default=None, help="Optional human-readable suffix for the run id."
    )
    p.add_argument(
        "--resume", type=Path, default=None, help="Path to an existing run directory to resume."
    )
    p.add_argument("--no-train", action="store_true", help="Run self-play only.")
    p.add_argument("--no-selfplay", action="store_true", help="Run trainer only.")
    p.add_argument(
        "--no-foreground",
        action="store_true",
        help="Even when only one side runs, redirect its output to a logfile "
        "instead of attaching to the terminal.",
    )
    p.add_argument(
        "--shutdown-grace",
        type=float,
        default=15.0,
        help="Seconds to wait for children to exit before SIGKILL.",
    )
    return p.parse_args()


def _make_run_dir(name: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)
    parts = [ts, suffix]
    if name:
        parts.append(name)
    run_dir = RUNS_ROOT / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in SUBDIRS:
        (run_dir / sub).mkdir()
    return run_dir


def _latest_step_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    return ckpts[-1] if ckpts else None


def _build_initial_engine(run_dir: Path, base_config: Path) -> Path:
    """Save a random-init checkpoint of the configured model, export it with
    scripts/build_net.sh, and seed the run's .iteration file."""
    import torch  # local import; pulls torch only when launching a fresh run.

    from src.models.factory import build_model
    from src.train.config import load_config

    init_pt = run_dir / "checkpoints" / "model_dynamic_initial.pt"
    cfg = load_config(base_config)
    model = build_model(cfg.model).eval()
    torch.save(
        {
            "model": model.state_dict(),
            "model_spec": model.spec(),
            "config": {"search_heads": cfg.search_heads},
        },
        init_pt,
    )

    engine_dir = run_dir / "engines"
    iter_file = engine_dir / ".iteration"

    print(f"[launcher] building the initial network ({init_pt} -> {engine_dir})", flush=True)
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "build_net.sh"),
            str(init_pt),
            "0",
            str(engine_dir),
            str(iter_file),
        ],
        cwd=ROOT,
        check=True,
    )
    initial = engine_dir / "model_dynamic_initial.cnet"
    if not initial.exists():
        raise RuntimeError(f"build_net.sh did not produce {initial}")
    return initial


def _has_engine(run_dir: Path) -> bool:
    """Whether the last engine engines/.iteration names is an existing .cnet network
    (runs from before the fused network list TensorRT engines, which no longer load)."""
    it = run_dir / "engines" / ".iteration"
    if not it.is_file():
        return False
    rows = [
        line
        for line in it.read_text().splitlines()
        if line.strip() and not line.startswith("iteration")
    ]
    if not rows:
        return False
    path = Path(rows[-1].split(",", 1)[1].strip())
    return path.suffix == ".cnet" and path.is_file()


def _build_resume_engine(run_dir: Path, ckpt: Path) -> None:
    """A branch run (another run's checkpoint copied in) has no engine yet, and a run from before
    the fused network has only TensorRT ones: self-play would wait for one forever while the
    trainer waits for self-play's games. Build it from the checkpoint."""
    step = ckpt.stem.split("_")[1]
    print(f"[launcher] no engine yet: building one from {ckpt.name}", flush=True)
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "build_net.sh"),
            str(ckpt),
            str(int(step)),
            str(run_dir / "engines"),
            str(run_dir / "engines" / ".iteration"),
        ],
        cwd=ROOT,
        check=True,
    )


def _write_run_config(base_config: Path, run_dir: Path, resume_from: Path | None) -> Path:
    """Load the base YAML, override every path field to point inside `run_dir`,
    write the result to <run_dir>/config.yaml, and return its path."""
    from src.train.config import load_yaml

    cfg = load_yaml(base_config)  # `extends:` chains resolved; the snapshot is self-contained
    cfg.setdefault("data", {})["dataset_path"] = str((run_dir / "buffer" / "buffer.bin").resolve())
    ckpt = cfg.setdefault("checkpoint", {})
    ckpt["dir"] = str((run_dir / "checkpoints").resolve())
    ckpt["engine_dir"] = str((run_dir / "engines").resolve())
    ckpt["iteration_file"] = str((run_dir / "engines" / ".iteration").resolve())
    sp = cfg.setdefault("selfplay", {})
    sp["pgn_dir"] = str((run_dir / "pgns").resolve())
    wb = cfg.setdefault("wandb", {})
    wb["dir"] = str((run_dir / "wandb").resolve())
    if resume_from:
        cfg["resume_path"] = str(resume_from.resolve())
        # Carry over wandb.resume_id from a previously written in-run config so
        # repeated resumes keep appending to the same W&B run.
        existing = run_dir / "config.yaml"
        if existing.is_file():
            prev = yaml.safe_load(existing.read_text()) or {}
            prev_id = (prev.get("wandb") or {}).get("resume_id")
            if prev_id and not wb.get("resume_id"):
                wb["resume_id"] = prev_id
    out = run_dir / "config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def _open_log(path: Path):
    fh = path.open("a", buffering=1)
    fh.write(f"\n========== launched {datetime.now().isoformat(timespec='seconds')} ==========\n")
    return fh


def _spawn(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    log_fh = _open_log(log_path)
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def mps_child_env(
    base: dict[str, str], pipe_dir: str, log_dir: str, trainer: bool
) -> dict[str, str]:
    """Environment of a process under the run's MPS daemon; the trainer at below-normal priority."""
    env = dict(base, CUDA_MPS_PIPE_DIRECTORY=pipe_dir, CUDA_MPS_LOG_DIRECTORY=log_dir)
    if trainer:
        env["CUDA_MPS_CLIENT_PRIORITY"] = "1"
    else:
        env.pop("CUDA_MPS_CLIENT_PRIORITY", None)
    return env


def _start_mps(trainer_priority: str) -> tuple[str, str] | None:
    """Start an MPS daemon with its own pipe directory (other GPU processes on the box keep
    their own contexts) and check that a client can connect. Returns (pipe_dir, log_dir), or
    None after a warning when MPS is unavailable."""
    control = shutil.which("nvidia-cuda-mps-control")
    if control is None:
        print("[launcher] mps: nvidia-cuda-mps-control not found; running without MPS", flush=True)
        return None
    base = Path(f"/tmp/chessrl-mps-{os.getpid()}")  # short: the pipe path has a length limit
    pipe_dir, log_dir = str(base / "pipe"), str(base / "log")
    Path(pipe_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    env = mps_child_env(dict(os.environ), pipe_dir, log_dir, trainer=False)
    if subprocess.run([control, "-d"], env=env).returncode != 0:
        print("[launcher] mps: the daemon did not start; running without MPS", flush=True)
        return None
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; torch.zeros(1, device='cuda'); torch.cuda.synchronize()",
        ],
        env=env,
        capture_output=True,
        timeout=120,
    )
    if probe.returncode != 0:
        print("[launcher] mps: a test client failed; running without MPS", flush=True)
        _stop_mps((pipe_dir, log_dir))
        return None
    level = "below-normal" if trainer_priority == "low" else "normal"
    print(f"[launcher] mps: daemon up ({pipe_dir}), trainer at {level} priority", flush=True)
    return pipe_dir, log_dir


def _stop_mps(mps: tuple[str, str]) -> None:
    control = shutil.which("nvidia-cuda-mps-control")
    if control is None:
        return
    env = mps_child_env(dict(os.environ), mps[0], mps[1], trainer=False)
    subprocess.run([control], input=b"quit\n", env=env)


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

    if args.resume:
        run_dir = args.resume.resolve()
        if not run_dir.is_dir():
            print(f"--resume target does not exist: {run_dir}", file=sys.stderr)
            return 2
        for sub in SUBDIRS:
            (run_dir / sub).mkdir(exist_ok=True)
        latest = _latest_step_checkpoint(run_dir)
        if latest is not None:
            print(f"[launcher] resuming run {run_dir.name} from {latest.name}", flush=True)
            if not args.no_selfplay and not _has_engine(run_dir):
                _build_resume_engine(run_dir, latest)
        else:
            print(f"[launcher] resuming run {run_dir.name} (no step checkpoints yet)", flush=True)
        resume_path = latest
    else:
        run_dir = _make_run_dir(args.name)
        print(f"[launcher] new run -> {run_dir}", flush=True)
        if not args.no_selfplay:
            _build_initial_engine(run_dir, args.config or DEFAULT_BASE_CONFIG)
        else:
            # Trainer-only fresh run: still need an iteration file so the
            # trainer's later build_net.sh invocations append cleanly.
            (run_dir / "engines" / ".iteration").write_text("iteration,engine_path\n")
        resume_path = None

    # On --resume without --config, reuse the run's own config.yaml.
    if args.config is None:
        own = run_dir / "config.yaml"
        args.config = own if (args.resume and own.is_file()) else DEFAULT_BASE_CONFIG
    cfg_path = _write_run_config(args.config, run_dir, resume_path)
    selfplay_log = run_dir / "selfplay.log"
    trainer_log = run_dir / "training.log"

    plan: list[tuple[str, list[str], Path]] = []
    if not args.no_selfplay:
        cmd_sp = [sys.executable, "-m", "src.self_play", "--config", str(cfg_path)]
        plan.append(("selfplay", cmd_sp, selfplay_log))
    if not args.no_train:
        cmd_tr = [sys.executable, "-m", "src.train", "--config", str(cfg_path)]
        plan.append(("training", cmd_tr, trainer_log))

    # If only ONE side is running, exec into it so it inherits the launcher's
    # TTY — that way Rich (the self-play dashboard) renders normally instead of
    # being downgraded to plain text by stdout-to-logfile redirection.
    if len(plan) == 1 and not args.no_foreground:
        name, cmd, _ = plan[0]
        print(f"\nRun directory: {run_dir}")
        print(f"Resolved config: {cfg_path}")
        print(f"Foreground: {name} (Ctrl-C to stop)\n", flush=True)
        os.execvp(cmd[0], cmd)  # does not return

    from src.train.config import load_config

    run_cfg = load_config(cfg_path)
    mps = _start_mps(run_cfg.mps_trainer_priority) if len(plan) == 2 and run_cfg.mps else None
    low_trainer = run_cfg.mps_trainer_priority == "low"
    children: list[tuple[str, subprocess.Popen, Path]] = []
    for name, cmd, log in plan:
        env = None
        if mps is not None:
            env = mps_child_env(
                dict(os.environ), mps[0], mps[1], trainer=name == "training" and low_trainer
            )
        children.append((name, _spawn(cmd, log, env), log))

    print(f"\nRun directory: {run_dir}")
    print(f"Resolved config: {cfg_path}")
    print(f"Launched {len(children)} subprocess(es):")
    for name, proc, log in children:
        print(f"  - {name:<8} pid={proc.pid}  log={log}")
    print("\nTail logs with:")
    for _, _, log in children:
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
        if mps is not None:
            _stop_mps(mps)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
