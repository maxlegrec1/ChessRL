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
import subprocess
import sys
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
POLL_INTERVAL_SECONDS = 2 * 3600   # 2h


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_state():
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text())
    return {"brawled": []}


def _save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


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
    print(f"[{_ts()}] brawl: step {step:>7d} vs SF-{SF_ELO}, {NUM_GAMES} games"
          f"  log={log.name}", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/targets/x86_64-linux/lib"
    cmd = [
        "uv", "run", "python", "scripts/stockfish_match.py",
        "--engine", str(engine),
        "--num-games", str(NUM_GAMES),
        "--sf-elo", str(SF_ELO),
        "--out", str(pgn),
    ]
    with log.open("w") as fh:
        subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                       cwd=str(ROOT), check=False)
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
    print(f"[{_ts()}] bootstrapped state: skipped {len(cands)-1} pre-existing 5k "
          f"engines (last skipped: {cands[-2] if len(cands)>=2 else 'none'}); "
          f"first brawl target = step {cands[-1]}", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    print(f"[{_ts()}] sf1400_autobrawl daemon started  pid={os.getpid()}  "
          f"poll={POLL_INTERVAL_SECONDS/3600:.1f}h  sf_elo={SF_ELO}  "
          f"games/brawl={NUM_GAMES}", flush=True)
    _bootstrap_state_if_empty()

    try:
        while True:
            try:
                state = _load_state()
                cand = _find_latest_unbrawled(state)
                if cand is None:
                    print(f"[{_ts()}] no new 5k engine; sleeping "
                          f"{POLL_INTERVAL_SECONDS/3600:.1f}h", flush=True)
                else:
                    step, engine = cand
                    result = _brawl(step, engine)
                    state.setdefault("brawled", []).append(step)
                    state["brawled"] = sorted(set(state["brawled"]))
                    state.setdefault("results", {})[str(step)] = result
                    _save_state(state)
            except Exception as e:
                print(f"[{_ts()}] error: {e}\n{traceback.format_exc()}", flush=True)
            time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        if PID_FILE.is_file():
            PID_FILE.unlink()


if __name__ == "__main__":
    main()
