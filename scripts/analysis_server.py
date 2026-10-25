"""Analysis board server: a checkpoint's move suggestions and a CPU search, behind a chessboard UI.

    .venv/bin/python scripts/analysis_server.py --port 8510        # http://localhost:8510

Lists the checkpoints synced under $CHESSRL_RUNS_ROOT (default ~/chessrl-runs) plus
any --extra checkpoint directories, best-rated first. Runs entirely on the CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.engine import Net, analyse, end_reason  # noqa: E402

STATIC = Path(__file__).resolve().parent / "analysis" / "index.html"
GAMES_PAGE = Path(__file__).resolve().parent / "analysis" / "games.html"
MAX_SIMS = 3200
# Ladder ratings of the encoder arm's late checkpoints measured through the
# FP16-overflowing engine; re-rated with fixed engines (docs/NEXT_ROUND.md).
RERATED = {("encoder", 30000): 1947, ("encoder", 40000): 1892, ("encoder", 45500): 1989}


def run_name(run_dir: Path) -> str:
    parts = run_dir.name.split("_")
    return "_".join(parts[3:]) if len(parts) > 3 else run_dir.name


def ladder_elo(run_dir: Path, step: int) -> float | None:
    try:
        rows = [json.loads(l) for l in (run_dir / "ladder" / "ladder.jsonl").read_text().splitlines() if l.strip()]
    except (FileNotFoundError, ValueError):
        return None
    near = [r for r in rows if abs(r.get("step", -10**9) - step) <= 2500]
    return min(near, key=lambda r: abs(r["step"] - step))["elo"] if near else None


def discover(runs_root: Path, extra: list[Path]) -> list[dict]:
    found = []
    newest: dict[Path, int] = {}
    cks = sorted(runs_root.glob("*/*/checkpoints/step_*.pt"))
    for ck in cks:
        newest[ck.parent] = max(newest.get(ck.parent, 0), int(ck.stem.split("_")[1]))
    for ck in cks:
        run = ck.parent.parent
        name = run_name(run)
        if name == "baseline" or name.endswith("-v0"):  # main-branch checkpoint format
            continue
        step = int(ck.stem.split("_")[1])
        # The sync brings home every 500-step checkpoint of a live run: list every
        # 5k steps plus each run's newest, so the picker stays usable.
        if step % 5000 and step != newest[ck.parent] and (name, step) not in RERATED:
            continue
        elo = RERATED.get((name, step), ladder_elo(run, step))
        found.append({"path": str(ck), "run": f"{name} ({run.parent.name})", "step": step, "elo": elo})
    for d in extra:
        for ck in sorted(d.glob("**/step_*.pt")):
            step = int(ck.stem.split("_")[1])
            found.append({"path": str(ck), "run": "encoder (armE)", "step": step, "elo": RERATED.get(("encoder", step))})
    for m in found:
        m["id"] = m["path"]  # stable across rescans
        elo = f"{m['elo']:,.0f}" if m["elo"] is not None else "unrated"
        m["label"] = f"{m['run']} · step {m['step'] / 1000:g}k · {elo}"
    return sorted(found, key=lambda m: -(m["elo"] or 0))


class Engines:
    """Discovered checkpoints (rescanned on demand) and loaded nets, least-recently-used first out."""

    def __init__(self, runs_root: Path, extra: list[Path], threads: int, default_run: str = "", keep: int = 4) -> None:
        self.runs_root, self.extra, self.default_run = runs_root, extra, default_run
        self.models: dict[str, dict] = {}
        self.rescan()
        self.threads, self.keep = threads, keep
        self.cache: OrderedDict[str, Net] = OrderedDict()
        self.lock = threading.Lock()

    def rescan(self) -> list[dict]:
        found = discover(self.runs_root, [d for d in self.extra if d.exists()])
        # Preselect the default run's newest checkpoint (else the best-rated one).
        mine = [m for m in found if m["run"].split(" ")[0] == self.default_run]
        pick = max(mine, key=lambda m: m["step"]) if mine else (found[0] if found else None)
        for m in found:
            m["default"] = m is pick
        self.models = {m["id"]: m for m in found}
        return found

    def get(self, model_id: str) -> Net:
        if model_id not in self.models:
            raise ValueError(f"unknown model {model_id}")
        with self.lock:
            if model_id in self.cache:
                self.cache.move_to_end(model_id)
                return self.cache[model_id]
            net = Net(self.models[model_id]["path"], threads=self.threads)
            self.cache[model_id] = net
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
            return net


def position(fen: str, moves: list[str]) -> tuple[chess.Board, list[str]]:
    """The board after `moves` from `fen`, and those moves in SAN."""
    board = chess.Board(fen)
    sans = []
    for u in moves:
        m = chess.Move.from_uci(u)
        if m not in board.legal_moves:
            raise ValueError(f"illegal move {u} in {board.fen()}")
        sans.append(board.san(m))
        board.push(m)
    return board, sans


def board_state(board: chess.Board) -> dict:
    dests: dict[str, list[str]] = {}
    for m in board.legal_moves:
        dests.setdefault(chess.square_name(m.from_square), [])
        to = chess.square_name(m.to_square)
        if to not in dests[chess.square_name(m.from_square)]:
            dests[chess.square_name(m.from_square)].append(to)
    last = board.peek().uci() if board.move_stack else None
    return {"fen": board.fen(), "turn": "white" if board.turn else "black", "dests": dests,
            "check": board.is_check(), "last": last, "history": len(board.move_stack)}


def pgn_files(runs_root: Path) -> list[Path]:
    """Self-play game samples fetched next to the synced runs (<runs_root>/<box>/*.pgn), newest first."""
    return sorted(runs_root.glob("*/*.pgn"), key=lambda p: p.stat().st_mtime, reverse=True)


def read_games(path: Path) -> list[dict]:
    """Every game of a PGN file with the FEN after each ply, for a view-only board."""
    import chess.pgn

    games = []
    with path.open() as fh:
        while (game := chess.pgn.read_game(fh)) is not None:
            board = game.board()
            sans, fens, last, checks = [], [board.fen()], [None], [False]
            for mv in game.mainline_moves():
                sans.append(board.san(mv))
                board.push(mv)
                fens.append(board.fen())
                last.append([chess.square_name(mv.from_square), chess.square_name(mv.to_square)])
                checks.append(board.is_check())
            result = game.headers.get("Result", "*")
            end = "checkmate" if board.is_checkmate() else ("draw" if result == "1/2-1/2" else "resignation")
            games.append({"n": len(games) + 1, "result": result, "end": end, "plies": len(sans), "san": sans,
                          "fens": fens, "last": last, "checks": checks})
    return games


def make_handler(engines: Engines):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
            elif self.path in ("/games", "/games.html"):
                self._send(200, GAMES_PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/pgns":
                root = engines.runs_root
                self._json(200, [{"id": str(p.relative_to(root)), "label": f"{p.parent.name} / {p.name}"}
                                 for p in pgn_files(root)])
            elif self.path.startswith("/api/pgn?id="):
                from urllib.parse import unquote

                wanted = unquote(self.path.split("=", 1)[1])
                files = {str(p.relative_to(engines.runs_root)): p for p in pgn_files(engines.runs_root)}
                if wanted not in files:  # only the listed samples, never an arbitrary path
                    return self._json(404, {"error": f"unknown PGN {wanted}"})
                self._json(200, {"id": wanted, "games": read_games(files[wanted])})
            elif self.path == "/api/models":
                self._json(200, [{k: m[k] for k in ("id", "label", "run", "step", "elo", "default")} for m in engines.rescan()])
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/analyse":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                board, sans = position(req.get("fen") or chess.STARTING_FEN, req.get("moves") or [])
                sims = max(0, min(MAX_SIMS, int(req.get("sims", 0))))
                state = board_state(board)
                state["sans"] = sans
                # Over only by the rules (an actual threefold, not a claimable one): a
                # side that could repeat may still play on.
                reason = end_reason(board)
                if reason:
                    state.update(moves=[], best=None, wdl=None, sims=0, seconds=0, status=reason)
                    return self._json(200, state)
                net = engines.get(str(req.get("model")))
                res = analyse(net, board, sims)
                state.update(res)
                state["params"] = net.params
                self._json(200, state)
            except (ValueError, KeyError) as exc:
                self._json(400, {"error": str(exc)})

    return Handler


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8510)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--threads", type=int, default=8, help="torch CPU threads per search")
    p.add_argument("--runs-root", type=Path, default=Path(os.environ.get("CHESSRL_RUNS_ROOT", Path.home() / "chessrl-runs")))
    p.add_argument("--extra", type=Path, nargs="*", default=[ROOT / "artifacts" / "scratch" / "branch45500"])
    p.add_argument("--default-run", default="encfinal", help="preselect this run's newest checkpoint")
    a = p.parse_args()
    engines = Engines(a.runs_root, a.extra, a.threads, a.default_run)
    best = max(engines.models.values(), key=lambda m: m["elo"] or 0, default=None)
    print(f"{len(engines.models)} checkpoints; best: {best['label'] if best else '-'}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(engines)).serve_forever()


if __name__ == "__main__":
    main()
