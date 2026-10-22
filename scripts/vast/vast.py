#!/usr/bin/env python3
"""Home-side Vast.ai tooling for ChessRL runs (stdlib only; shells out to `vastai`).

  chessrl-vast offers [--gpu RTX_4090 RTX_5090] [--min-cores 32]
  chessrl-vast launch --offer ID --label NAME      # budget-checked
  chessrl-vast status                               # instances, spend, ssh, dashboard ports
  chessrl-vast ssh LABEL [cmd...]                   # run a command on a box
  chessrl-vast destroy LABEL
  chessrl-vast guard                                # spend check; destroys everything at the cap
  chessrl-vast tunnels                              # keep localhost:85xx -> box:8501 forwards up
  chessrl-vast sync                                 # pull metrics/logs/milestone checkpoints home

State lives in ~/.chessrl-vast/ (ledger.json, budget.json, known_hosts). The
Vast API key never leaves this machine; boxes only receive budget.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STATE = Path.home() / ".chessrl-vast"
LEDGER = STATE / "ledger.json"
BUDGET = STATE / "budget.json"
KNOWN_HOSTS = STATE / "known_hosts"
RUNS_HOME = Path.home() / "chessrl-runs"
CAP_USD = 200.0
LAUNCH_CAP_USD = 180.0
IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
BOX_RUNS = ("/workspace/ChessRL/artifacts/runs/", "/workspace/ChessRL-main/artifacts/runs/")
DASHBOARD_PORT = 8501


def vast(*args: str):
    out = subprocess.run(["vastai", *args, "--raw"], capture_output=True, text=True, check=True).stdout
    return json.loads(out) if out.strip() else None


def load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"instances": {}}


def save_ledger(ledger: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2))
    tmp.replace(LEDGER)


def active(ledger: dict) -> dict:
    return {k: v for k, v in ledger["instances"].items() if v.get("ended") is None}


def by_label(ledger: dict, label: str) -> tuple[str, dict]:
    # Prefer a live instance: a label can be reused after its box was destroyed.
    matches = [(iid, inst) for iid, inst in ledger["instances"].items() if inst["label"] == label or iid == label]
    live = [m for m in matches if m[1].get("ended") is None]
    if live or matches:
        return (live or matches)[-1]
    sys.exit(f"no instance labelled {label!r}")


def spend(ledger: dict, now: float | None = None) -> float:
    now = now or time.time()
    return sum(((i.get("ended") or now) - i["started"]) / 3600.0 * i["dph"] for i in ledger["instances"].values())


def burn_rate(ledger: dict) -> float:
    return sum(i["dph"] for i in active(ledger).values())


def ssh_target(iid: str) -> tuple[str, int] | None:
    info = vast("show", "instance", iid)
    if not info:
        return None
    ports = (info.get("ports") or {}).get("22/tcp")
    if info.get("public_ipaddr") and ports:
        return info["public_ipaddr"].strip(), int(ports[0]["HostPort"])
    if info.get("ssh_host") and info.get("ssh_port"):
        return info["ssh_host"], int(info["ssh_port"])
    return None


def ssh_base(target: tuple[str, int]) -> list[str]:
    host, port = target
    return [
        "ssh", "-p", str(port),
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15", f"root@{host}",
    ]


# --------------------------------------------------------------------------- #
def search_offers(gpus: list[str], min_cores: int, disk: int) -> list[dict]:
    rows = []
    for gpu in gpus:
        q = (f"gpu_name={gpu} num_gpus=1 cpu_cores_effective>={min_cores} cuda_max_good>=12.8 "
             f"reliability>0.98 verified=true rentable=true inet_down>=200 disk_space>={disk}")
        rows += vast("search", "offers", q, "-o", "dph") or []
    return sorted(rows, key=lambda o: o["dph_total"])


def cmd_offers(a) -> None:
    rows = search_offers(a.gpu, a.min_cores, a.disk)
    print(f"{'offer':>9} {'gpu':<10} {'$/h':>6} {'cores':>5} {'ram':>4} {'cuda':>5} {'dl':>5}  cpu / location")
    for o in rows[: a.n]:
        print(f"{o['id']:>9} {o['gpu_name']:<10} {o['dph_total']:6.3f} {o.get('cpu_cores_effective', 0):5.0f} "
              f"{(o.get('cpu_ram') or 0) // 1024:4d} {o.get('cuda_max_good', 0):5.1f} {o.get('inet_down', 0):5.0f}  "
              f"{(o.get('cpu_name') or '')[:26]} / {o.get('geolocation', '')}")


def cmd_launch(a) -> None:
    ledger = load_ledger()
    # `search offers id=N` matches nothing, so find the offer in the usual search.
    # Same filters as `offers` (a looser query hits the result cap before cheap 32-core boxes).
    rows = [o for o in search_offers(["RTX_4090", "RTX_5090"], 32, a.disk) if str(o["id"]) == str(a.offer)]
    if not rows:
        sys.exit(f"offer {a.offer} not available (run `offers` again)")
    offer = rows[0]
    projected = spend(ledger) + burn_rate(ledger) + offer["dph_total"]  # one more hour of everything
    if projected > LAUNCH_CAP_USD:
        sys.exit(f"refusing: projected spend ${projected:.2f} > launch cap ${LAUNCH_CAP_USD:.0f}")
    res = vast("create", "instance", str(a.offer), "--image", a.image, "--disk", str(a.disk),
               "--ssh", "--direct", "--label", a.label)
    iid = str(res["new_contract"])
    used = {i.get("local_port") for i in active(ledger).values()}
    port = next(p for p in range(DASHBOARD_PORT, DASHBOARD_PORT + 50) if p not in used)
    ledger["instances"][iid] = {
        "label": a.label, "offer": a.offer, "gpu": offer["gpu_name"], "dph": offer["dph_total"],
        "cores": offer.get("cpu_cores_effective"), "started": time.time(), "ended": None, "local_port": port,
        "deadline": time.time() + a.hours * 3600 if a.hours else None,
    }
    save_ledger(ledger)
    print(f"launched {a.label}: instance {iid} ({offer['gpu_name']}, ${offer['dph_total']:.3f}/h), "
          f"dashboard -> http://localhost:{port}" + (f", auto-destroy in {a.hours:g} h" if a.hours else ""))


def cmd_deadline(a) -> None:
    """Set (or clear with --clear) the time after which guard syncs and destroys a box."""
    ledger = load_ledger()
    iid, inst = by_label(ledger, a.label)
    if a.clear:
        inst["deadline"] = None
    elif a.at:
        from datetime import datetime

        inst["deadline"] = datetime.fromisoformat(a.at.replace("Z", "+00:00")).timestamp()
    else:
        inst["deadline"] = time.time() + a.hours * 3600
    save_ledger(ledger)
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(inst["deadline"])) if inst["deadline"] else "none"
    print(f"{inst['label']}: deadline {when}")


def cmd_status(a) -> None:
    ledger = load_ledger()
    print(f"spent ${spend(ledger):.2f} / ${CAP_USD:.0f}   burn ${burn_rate(ledger):.2f}/h")
    for iid, inst in active(ledger).items():
        hours = (time.time() - inst["started"]) / 3600
        target = ssh_target(iid)
        ssh = " ".join(ssh_base(target)[:3] + [ssh_base(target)[-1]]) if target else "(no ssh yet)"
        print(f"  {inst['label']:<14} id={iid} {inst['gpu']} ${inst['dph']:.3f}/h {hours:5.1f}h "
              f"dash=http://localhost:{inst['local_port']}  {ssh}")


def cmd_ssh(a) -> None:
    ledger = load_ledger()
    iid, _ = by_label(ledger, a.label)
    target = ssh_target(iid)
    if not target:
        sys.exit("ssh not ready")
    os.execvp("ssh", ssh_base(target) + a.cmd)


def cmd_bootstrap(a) -> None:
    """Wait for ssh, then run scripts/vast/setup_box.sh (fetched by commit SHA) in the background."""
    ledger = load_ledger()
    iid, inst = by_label(ledger, a.label)
    out = subprocess.run(["git", "ls-remote", "https://github.com/maxlegrec1/ChessRL.git",
                          f"refs/heads/{a.branch}"], capture_output=True, text=True, check=True).stdout
    sha = out.split()[0]
    url = f"https://raw.githubusercontent.com/maxlegrec1/ChessRL/{sha}/scripts/vast/setup_box.sh"
    deadline = time.time() + a.timeout
    while time.time() < deadline:
        target = None
        try:
            target = ssh_target(iid)
        except Exception:
            pass
        if target:
            r = subprocess.run(ssh_base(target) + [
                f"mkdir -p /workspace && cd /workspace && curl -fsSL {url} -o setup_box.sh && "
                f"(nohup bash setup_box.sh {a.branch} > setup.log 2>&1 &) && echo started"],
                capture_output=True, text=True)
            if "started" in r.stdout:
                print(f"{inst['label']}: setup started from {sha[:8]} (tail /workspace/setup.log)")
                return
        time.sleep(20)
    sys.exit(f"{inst['label']}: ssh not ready after {a.timeout}s")


def destroy(ledger: dict, iid: str) -> None:
    subprocess.run(["vastai", "destroy", "instance", iid], capture_output=True, text=True)
    ledger["instances"][iid]["ended"] = time.time()
    save_ledger(ledger)


def cmd_destroy(a) -> None:
    ledger = load_ledger()
    iid, inst = by_label(ledger, a.label)
    destroy(ledger, iid)
    print(f"destroyed {inst['label']} ({iid}); total spend ${spend(ledger):.2f}")


def cmd_guard(a) -> None:
    ledger = load_ledger()
    # Refresh prices (actual dph includes the disk we asked for) and notice
    # instances that disappeared on Vast's side.
    live = {str(i["id"]): i for i in (vast("show", "instances") or [])}
    for iid, inst in active(ledger).items():
        if iid in live:
            inst["dph"] = float(live[iid].get("dph_total") or inst["dph"])
        else:
            inst["ended"] = time.time()
    save_ledger(ledger)
    for iid, inst in list(active(ledger).items()):
        if inst.get("deadline") and time.time() >= inst["deadline"]:
            sync_one(iid, inst)  # final pull before the box disappears
            destroy(ledger, iid)
            print(f"deadline reached: synced and destroyed {inst['label']} ({iid})")
    spent = spend(ledger)
    if spent >= CAP_USD:
        for iid in list(active(ledger)):
            destroy(ledger, iid)
        print(f"BUDGET CAP REACHED (${spent:.2f}): destroyed all instances")
    rate = burn_rate(ledger)
    budget = {
        "t": time.time(), "spent": round(spent, 3), "cap": CAP_USD, "launch_cap": LAUNCH_CAP_USD,
        "burn_per_hour": round(rate, 3), "hours_left": round((CAP_USD - spent) / rate, 1) if rate else None,
        "instances": [
            {"label": i["label"], "gpu": i["gpu"], "dph": i["dph"], "started": i["started"], "ended": i["ended"]}
            for i in ledger["instances"].values()
        ],
    }
    STATE.mkdir(parents=True, exist_ok=True)
    BUDGET.write_text(json.dumps(budget, indent=2))
    for iid in active(ledger):
        target = ssh_target(iid)
        if target:
            host, port = target
            subprocess.run(
                ["scp", "-q", "-P", str(port), "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
                 "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
                 str(BUDGET), f"root@{host}:/workspace/budget.json"],
                capture_output=True,
            )
    print(f"spent ${spent:.2f} / ${CAP_USD:.0f}, burn ${rate:.2f}/h")


def cmd_tunnels(a) -> None:
    procs: dict[str, tuple[subprocess.Popen, tuple[str, int]]] = {}
    while True:
        ledger = load_ledger()
        live = active(ledger)
        for iid in list(procs):
            if iid not in live:
                procs.pop(iid)[0].terminate()
        for iid, inst in live.items():
            proc = procs.get(iid)
            if proc and proc[0].poll() is None:
                continue
            try:
                target = ssh_target(iid)
            except Exception:
                target = None
            if not target:
                continue
            cmd = ssh_base(target)
            cmd[1:1] = ["-N", "-L", f"{inst['local_port']}:localhost:{DASHBOARD_PORT}", "-o", "ExitOnForwardFailure=yes"]
            procs[iid] = (subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), target)
            print(f"tunnel {inst['label']}: localhost:{inst['local_port']} -> {target[0]}:{target[1]}", flush=True)
        time.sleep(15)


def sync_one(iid: str, inst: dict) -> None:
    try:
        target = ssh_target(iid)
    except Exception:
        target = None
    if not target:
        return
    host, port = target
    dest = RUNS_HOME / inst["label"]
    dest.mkdir(parents=True, exist_ok=True)
    ssh = f"ssh -p {port} -o UserKnownHostsFile={KNOWN_HOSTS} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
    filters = ["--include=*/", "--include=metrics/***", "--include=ladder/***", "--include=config.yaml",
               "--include=*.log", "--include=*.stats.json", "--include=step_*0000.pt", "--exclude=*"]
    for src in BOX_RUNS:
        r = subprocess.run(["rsync", "-az", "--prune-empty-dirs", *filters, "-e", ssh,
                            f"root@{host}:{src}", f"{dest}/"], capture_output=True, text=True)
        if r.returncode not in (0, 23):  # 23: source dir absent on this box
            print(f"sync {inst['label']} {src}: rc={r.returncode} {r.stderr.strip()[:200]}")


def cmd_sync(a) -> None:
    ledger = load_ledger()
    for iid, inst in active(ledger).items():
        sync_one(iid, inst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("offers")
    o.add_argument("--gpu", nargs="+", default=["RTX_4090", "RTX_5090"])
    o.add_argument("--min-cores", type=int, default=32)
    o.add_argument("--disk", type=int, default=120)
    o.add_argument("-n", type=int, default=15)
    o.set_defaults(fn=cmd_offers)
    la = sub.add_parser("launch")
    la.add_argument("--offer", required=True)
    la.add_argument("--label", required=True)
    la.add_argument("--image", default=IMAGE)
    la.add_argument("--disk", type=int, default=120)
    la.add_argument("--hours", type=float, default=0, help="auto sync+destroy after this many hours (0 = never)")
    la.set_defaults(fn=cmd_launch)
    dl = sub.add_parser("deadline")
    dl.add_argument("label")
    dl.add_argument("--hours", type=float, default=0)
    dl.add_argument("--at", help="ISO time, e.g. 2026-10-22T20:50Z")
    dl.add_argument("--clear", action="store_true")
    dl.set_defaults(fn=cmd_deadline)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("ssh")
    s.add_argument("label")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_ssh)
    b = sub.add_parser("bootstrap")
    b.add_argument("label")
    b.add_argument("--branch", default="worktree-next-round")
    b.add_argument("--timeout", type=int, default=1200)
    b.set_defaults(fn=cmd_bootstrap)
    d = sub.add_parser("destroy")
    d.add_argument("label")
    d.set_defaults(fn=cmd_destroy)
    sub.add_parser("guard").set_defaults(fn=cmd_guard)
    sub.add_parser("tunnels").set_defaults(fn=cmd_tunnels)
    sub.add_parser("sync").set_defaults(fn=cmd_sync)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
