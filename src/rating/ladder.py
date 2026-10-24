"""Anchor ladder: fixed opponents, batched MCTS games, Elo fits, puzzle accuracy.

Anchors span "random mover" to node-limited Stockfish. Node limits (not time)
keep the yardstick identical on any CPU, so runs on different rented boxes are
comparable. Anchor ratings come from a one-off round robin (`calibrate`), on a
relative scale with the random mover at 0; a model's rating is the
maximum-likelihood fit of its results against anchors with those fixed ratings.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RATINGS = ROOT / "src" / "rating" / "anchors.json"
PUZZLES = ROOT / "data" / "puzzles" / "lichess_1k.csv"

_PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

# name -> (kind, params). Ordered weakest to strongest (by design; calibration decides).
ANCHORS: dict[str, tuple[str, dict]] = {
    "random": ("random", {}),
    "greedy": ("greedy", {}),
    # Skill 0 caps Stockfish's depth, so node counts below ~50 barely change its
    # strength (1/10/50 nodes all calibrated within 50 Elo). Mixing in uniformly
    # random moves fills the gap between greedy and skill-0 Stockfish instead.
    "sf_s0_r50": ("stockfish", {"skill": 0, "nodes": 50, "random_prob": 0.5}),
    "sf_s0_r25": ("stockfish", {"skill": 0, "nodes": 50, "random_prob": 0.25}),
    "sf_s0_n50": ("stockfish", {"skill": 0, "nodes": 50}),
    "sf_s5_n300": ("stockfish", {"skill": 5, "nodes": 300}),
    "sf_s10_n1000": ("stockfish", {"skill": 10, "nodes": 1000}),
    "sf_s20_n1000": ("stockfish", {"skill": 20, "nodes": 1000}),
    "sf_s20_n5000": ("stockfish", {"skill": 20, "nodes": 5000}),
    "sf_s20_n20000": ("stockfish", {"skill": 20, "nodes": 20000}),
}


def stockfish_path() -> str:
    import shutil

    # Debian/Ubuntu's package installs to /usr/games, often not on PATH.
    return shutil.which("stockfish") or "/usr/games/stockfish"


class Anchor:
    """A fixed opponent. Stockfish anchors keep one engine process each."""

    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self.kind, self.params = ANCHORS[name]
        self.rng = random.Random(seed)
        self.engine = None
        if self.kind == "stockfish":
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path())
            self.engine.configure({"Skill Level": self.params["skill"], "Threads": 1, "Hash": 16})

    def move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if self.kind == "random":
            return self.rng.choice(legal)
        if self.kind == "greedy":
            # Mate if possible, else the most valuable capture, else random.
            best, best_score = [], -1.0
            for m in legal:
                board.push(m)
                mate = board.is_checkmate()
                board.pop()
                if mate:
                    return m
                score = _PIECE_VALUE[board.piece_type_at(m.to_square)] if board.is_capture(m) and \
                    board.piece_type_at(m.to_square) else 0.0
                if score > best_score:
                    best, best_score = [m], score
                elif score == best_score:
                    best.append(m)
            return self.rng.choice(best)
        if self.rng.random() < self.params.get("random_prob", 0.0):
            return self.rng.choice(legal)
        result = self.engine.play(board, chess.engine.Limit(nodes=self.params["nodes"]))
        return result.move

    def close(self) -> None:
        if self.engine is not None:
            self.engine.quit()


@dataclass
class Game:
    opening: list[str]
    model_white: bool
    board: chess.Board = field(default_factory=chess.Board)
    history: list[str] = field(default_factory=list)
    result: float | None = None  # score from the model's point of view

    def __post_init__(self) -> None:
        for uci in self.opening:
            self.board.push_uci(uci)
            self.history.append(uci)

    @property
    def model_to_move(self) -> bool:
        return (self.board.turn == chess.WHITE) == self.model_white


def rules_end(board: chess.Board) -> tuple[bool, chess.Color | None]:
    """(over, winner) under automatic draws, the same rules as the self-play referee:
    checkmate, stalemate, insufficient material, the 50-move rule, or a position
    occurring for the third time. Unlike python-chess's claim_draw, a threefold that
    could be made with the next move does not end the game."""
    if board.is_checkmate():
        return True, not board.turn
    if (board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100
            or board.is_repetition(3)):
        return True, None
    return False, None


def make_openings(n: int, plies: int = 8, seed: int = 1234) -> list[list[str]]:
    """Deterministic random openings (each is played twice, colours swapped)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        moves = []
        for _ in range(plies):
            legal = list(b.legal_moves)
            if not legal:
                break
            m = rng.choice(legal)
            b.push(m)
            moves.append(m.uci())
        if not b.is_game_over():
            out.append(moves)
    return out


def mcts_moves(engine_path: str, games: list[Game], sims: int) -> list[str]:
    import _inference_cpp

    res = _inference_cpp.small_mcts_search_batch_with_history(
        histories=[g.history for g in games],
        fens=[""] * len(games),
        simulations=sims,
        cpuct=0.65,
        temperature=1.0,
        engine_path=engine_path,
        temperature_at_root=0.0,
        use_dirichlet=False,
        virtual_loss_width=1,
    )
    return [r["action"] for r in res]


def play_match(engine_path: str, anchor: Anchor, openings: list[list[str]], sims: int,
               max_plies: int = 300) -> list[float]:
    """Model vs anchor, every opening with both colours. Returns model scores."""
    games = [Game(o, w) for o in openings for w in (True, False)]
    while True:
        live = [g for g in games if g.result is None]
        if not live:
            break
        for g in live:
            over, winner = rules_end(g.board)
            if over or len(g.history) >= max_plies:
                g.result = 0.5 if winner is None else (1.0 if winner == g.model_white else 0.0)
        live = [g for g in games if g.result is None]
        to_search = [g for g in live if g.model_to_move]
        if to_search:
            for g, uci in zip(to_search, mcts_moves(engine_path, to_search, sims), strict=True):
                g.board.push_uci(uci)
                g.history.append(uci)
        for g in live:
            if g.result is None and not g.model_to_move and not rules_end(g.board)[0]:
                m = anchor.move(g.board)
                g.board.push(m)
                g.history.append(m.uci())
    return [g.result for g in games]


def expected(r: float, r_opp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_opp - r) / 400.0))


def fit_rating(results: list[tuple[float, float]], prior: float = 0.0) -> tuple[float, float]:
    """MLE rating from (opponent_rating, score) pairs; returns (rating, std error).

    A weak prior (two virtual draws against `prior`) keeps all-win / all-loss
    results finite."""
    data = list(results) + [(prior, 0.5), (prior, 0.5)]
    r = prior
    for _ in range(60):
        grad = hess = 0.0
        for opp, s in data:
            p = expected(r, opp)
            grad += (s - p)
            hess -= p * (1.0 - p)
        k = math.log(10.0) / 400.0
        step = -(grad * k) / (hess * k * k) if hess else 0.0
        r += max(-400.0, min(400.0, step))
        if abs(step) < 0.01:
            break
    info = sum(expected(r, o) * (1 - expected(r, o)) for o, _ in data) * (math.log(10.0) / 400.0) ** 2
    return r, (1.0 / math.sqrt(info)) if info > 0 else float("inf")


def load_anchor_ratings() -> dict[str, float]:
    if ANCHOR_RATINGS.exists():
        return json.loads(ANCHOR_RATINGS.read_text())["ratings"]
    return {}


def pick_anchors(estimate: float | None, ratings: dict[str, float], k: int = 2) -> list[str]:
    """The k calibrated anchors closest to the current estimate (bracketing it if possible)."""
    names = sorted(ratings, key=ratings.get)
    if estimate is None or not names:
        return names[:k] or ["random", "greedy"]
    below = [n for n in names if ratings[n] <= estimate]
    above = [n for n in names if ratings[n] > estimate]
    picks = below[-(k // 2):] + above[: k - k // 2]
    if len(picks) < k:
        picks = sorted(names, key=lambda n: abs(ratings[n] - estimate))[:k]
    return picks


def load_puzzles(path: Path = PUZZLES) -> list[dict]:
    with path.open(newline="") as fh:
        return [
            {"id": r["PuzzleId"], "fen": r["FEN"], "moves": r["Moves"].split(), "rating": int(r["Rating"])}
            for r in csv.DictReader(fh)
        ]


def puzzle_accuracy(engine_path: str, puzzles: list[dict], sims: int) -> dict:
    """Strict line play: after the puzzle's first (opponent) move, every model move must match."""
    import _inference_cpp

    state = [{"p": p, "ply": 1, "ok": True, "first": None} for p in puzzles]
    while True:
        live = [s for s in state if s["ok"] and s["ply"] < len(s["p"]["moves"])]
        if not live:
            break
        res = _inference_cpp.small_mcts_search_batch_with_history(
            histories=[s["p"]["moves"][: s["ply"]] for s in live],
            fens=[s["p"]["fen"] for s in live],
            simulations=sims, cpuct=0.65, temperature=1.0, engine_path=engine_path,
            temperature_at_root=0.0, use_dirichlet=False, virtual_loss_width=1,
        )
        for s, r in zip(live, res, strict=True):
            want = s["p"]["moves"][s["ply"]]
            good = r["action"] == want
            if s["first"] is None:
                s["first"] = good
            s["ok"] = good
            s["ply"] += 2  # skip the opponent's scripted reply
    bands: dict[str, list[int]] = {}
    for s in state:
        lo = (s["p"]["rating"] // 400) * 400
        bands.setdefault(f"{lo}-{lo + 399}", []).append(int(s["ok"]))
    n = len(state)
    return {
        "puzzles": n,
        "first_move": sum(bool(s["first"]) for s in state) / n,
        "line": sum(s["ok"] for s in state) / n,
        "line_by_band": {k: sum(v) / len(v) for k, v in sorted(bands.items())},
    }


def _play_pair(a_name: str, b_name: str, openings: list[list[str]]) -> list[tuple[str, str, float]]:
    a, b = Anchor(a_name, seed=1), Anchor(b_name, seed=2)
    out = []
    try:
        for opening in openings:
            for a_white in (True, False):
                board = chess.Board()
                for uci in opening:
                    board.push_uci(uci)
                while not rules_end(board)[0] and board.ply() < 300:
                    mover = a if (board.turn == chess.WHITE) == a_white else b
                    board.push(mover.move(board))
                winner = rules_end(board)[1]
                s = 0.5 if winner is None else (1.0 if winner == a_white else 0.0)
                out.append((a_name, b_name, s))
    finally:
        a.close()
        b.close()
    print(f"{a_name} vs {b_name}: {sum(s for *_, s in out):.1f}/{len(out)}", flush=True)
    return out


def calibrate(games_per_pair: int = 100, out: Path = ANCHOR_RATINGS) -> dict[str, float]:
    """Round robin between neighbouring anchors (plus a skip-one pair) -> ratings, random = 0."""
    from concurrent.futures import ProcessPoolExecutor

    names = list(ANCHORS)
    openings = make_openings(games_per_pair // 2, seed=99)
    pairs = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
    pairs += [(names[i], names[i + 2]) for i in range(len(names) - 2)]
    results: list[tuple[str, str, float]] = []  # (a, b, score of a)
    with ProcessPoolExecutor(max_workers=len(pairs)) as pool:
        for chunk in pool.map(_play_pair, *zip(*pairs, strict=True), [openings] * len(pairs)):
            results += chunk
    # Joint Bradley-Terry fit by coordinate ascent; random pinned at 0.
    ratings = {n: 200.0 * i for i, n in enumerate(names)}
    for _ in range(200):
        for n in names[1:]:
            vs = [(ratings[b], s) for a, b, s in results if a == n] + \
                 [(ratings[a], 1 - s) for a, b, s in results if b == n]
            ratings[n] = fit_rating(vs, prior=ratings[n])[0]
    ratings = {n: round(r - ratings["random"], 1) for n, r in ratings.items()}
    out.write_text(json.dumps({"ratings": ratings, "games_per_pair": games_per_pair}, indent=2))
    return ratings
