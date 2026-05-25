"""Autonomous SF-1400 brawler.

Polls every POLL_INTERVAL_SECONDS for a new 5k-multiple checkpoint engine
in the active run; if one exists that hasn't been brawled yet, brawls it
50 games vs Stockfish-1400 (MCTS-800, temp_root=0 — Max's defaults).

State (artifacts/exp/sf1400_autobrawl/state.json) tracks {step → result_line}.
PGNs land in artifacts/pgns/sf_1400_vs_step_NNNNNNNN_auto.pgn.
Per-brawl logs in artifacts/exp/sf1400_autobrawl/logs/step_NNNNNNNN.log.

Run with nohup. PID file at artifacts/exp/sf1400_autobrawl/daemon.pid.
Stop with: kill $(cat artifacts/exp/sf1400_autobrawl/daemon.pid)
"""

import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts/runs/20260515_133813_99ec_history_fix_v1"
ENGINES_DIR = RUN_DIR / "engines"
OUT_DIR = ROOT / "artifacts/exp/sf1400_autobrawl"
PGN_DIR = ROOT / "artifacts/pgns"
STATE_FILE = OUT_DIR / "state.json"
PID_FILE = OUT_DIR / "daemon.pid"
LOG_DIR = OUT_DIR / "logs"

STEP_BIN = 5000
SF_ELO = 1400
NUM_GAMES = 50
POLL_INTERVAL_SECONDS = 2 * 3600  # 2h

WANDB_RUN_ID_FILE = RUN_DIR / "wandb_run_id.txt"
WANDB_PROJECT = "self_play"  # mirrors src/train/config.py default

# "Model vs Stockfish-1400: 17W / 4D / 29L  ->  19.0/50  (38.0%)"
_RESULT_RE = re.compile(r"(\d+)W\s*/\s*(\d+)D\s*/\s*(\d+)L\s*->\s*(\d+\.\d+)/(\d+)\s*\(([\d.]+)%\)")


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_state():
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text())
    return {"brawled": []}


def _save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _push_to_wandb(step: int, result_line: str) -> None:
    """Attach to the trainer's W&B run and log this brawl's win-rate / Elo.

    Fail-soft: if wandb_run_id.txt is missing (older training run without the
    Phase-3 instrumentation), or wandb is unreachable, we just continue. The
    JSON state.json sidecar is still the source of truth.
    """
    if not WANDB_RUN_ID_FILE.is_file():
        return
    m = _RESULT_RE.search(result_line)
    if not m:
        print(f"[wandb] could not parse '{result_line}'; skipping push", flush=True)
        return
    pct = float(m.group(6))
    p = pct / 100.0
    # Symmetric Elo around 1400 (clamped to ±800 at endpoints to avoid log(0)).
    import math

    if 0.0 < p < 1.0:
        implied_elo = SF_ELO + (-400.0 * math.log10((1 - p) / p))
    else:
        implied_elo = SF_ELO + (800.0 if p >= 1.0 else -800.0)
    rid = WANDB_RUN_ID_FILE.read_text().strip()
    try:
        import wandb

        wandb.init(
            project=WANDB_PROJECT,
            id=rid,
            resume="must",
            reinit=True,
            settings=wandb.Settings(silent=True),
        )
        wandb.log(
            {
                "eval/sf1400_win_pct": pct,
                "eval/sf1400_implied_elo": implied_elo,
            },
            step=step,
        )
        wandb.finish()
        print(
            f"[wandb] logged sf1400/win_pct={pct:.1f} elo={implied_elo:.0f} at step {step}",
            flush=True,
        )
    except Exception as exc:
        print(f"[wandb] log failed (non-fatal): {exc}", flush=True)


def _find_latest_unbrawled(state):
    brawled = set(state.get("brawled", []))
    cands = []
    for e in sorted(ENGINES_DIR.glob("step_*.trt")):
        try:
            step = int(e.stem.split("_")[1])
        except Exception:
            continue
        if step % STEP_BIN != 0:
            continue
        if step in brawled:
            continue
        if e.stat().st_size == 0:
            continue
        cands.append((step, e))
    return cands[-1] if cands else None


def _brawl(step, engine):
    pgn = PGN_DIR / f"sf_{SF_ELO}_vs_step_{step:08d}_auto.pgn"
    log = LOG_DIR / f"step_{step:08d}.log"
    print(
        f"[{_ts()}] brawl: step {step:>7d} vs SF-{SF_ELO}, {NUM_GAMES} games  log={log.name}",
        flush=True,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/targets/x86_64-linux/lib"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/stockfish_match.py",
        "--engine",
        str(engine),
        "--num-games",
        str(NUM_GAMES),
        "--sf-elo",
        str(SF_ELO),
        "--out",
        str(pgn),
    ]
    with log.open("w") as fh:
        subprocess.run(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), check=False
        )
    # Parse the "Model vs Stockfish-NNNN: ..." final line
    final_line = ""
    for ln in log.read_text().splitlines():
        if "Model vs Stockfish" in ln and "->" in ln:
            final_line = ln.strip()
            break
    print(f"[{_ts()}] done : step {step:>7d}  {final_line}", flush=True)
    return final_line


def _bootstrap_state_if_empty():
    """First run: mark all CURRENT 5k engines except the latest as already
    brawled, so the daemon starts at the most recent checkpoint and only goes
    forward from there. The latest 5k engine is left for the first poll."""
    if STATE_FILE.is_file():
        return
    cands = []
    for e in sorted(ENGINES_DIR.glob("step_*.trt")):
        try:
            step = int(e.stem.split("_")[1])
        except Exception:
            continue
        if step % STEP_BIN != 0:
            continue
        cands.append(step)
    if not cands:
        return
    # All but the latest considered "already done" → the latest will be brawled first
    state = {
        "brawled": sorted(cands[:-1]),
        "results": {},
        "bootstrapped_at": _ts(),
        "bootstrap_skip_steps": cands[:-1],
        "comment": "All 5k engines except the latest are marked done so this "
        "daemon starts at the most-recent checkpoint and goes forward.",
    }
    _save_state(state)
    print(
        f"[{_ts()}] bootstrapped state: skipped {len(cands) - 1} pre-existing 5k "
        f"engines (last skipped: {cands[-2] if len(cands) >= 2 else 'none'}); "
        f"first brawl target = step {cands[-1]}",
        flush=True,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    print(
        f"[{_ts()}] sf1400_autobrawl daemon started  pid={os.getpid()}  "
        f"poll={POLL_INTERVAL_SECONDS / 3600:.1f}h  sf_elo={SF_ELO}  "
        f"games/brawl={NUM_GAMES}",
        flush=True,
    )
    _bootstrap_state_if_empty()

    try:
        while True:
            try:
                state = _load_state()
                cand = _find_latest_unbrawled(state)
                if cand is None:
                    print(
                        f"[{_ts()}] no new 5k engine; sleeping {POLL_INTERVAL_SECONDS / 3600:.1f}h",
                        flush=True,
                    )
                else:
                    step, engine = cand
                    result = _brawl(step, engine)
                    state.setdefault("brawled", []).append(step)
                    state["brawled"] = sorted(set(state["brawled"]))
                    state.setdefault("results", {})[str(step)] = result
                    _save_state(state)
                    _push_to_wandb(step, result)
            except Exception as e:
                print(f"[{_ts()}] error: {e}\n{traceback.format_exc()}", flush=True)
            time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        if PID_FILE.is_file():
            PID_FILE.unlink()


if __name__ == "__main__":
    main()
