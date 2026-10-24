"""CPU analysis engine: a checkpoint's policy/value heads plus a PUCT search.

Mirrors the self-play engine's conventions (cpp/mcts/mcts_small.cpp): absolute
111-plane input with 7 plies of history and repetition flags, UCI policy
vocabulary (castling as the king's move, knight promotions as the bare move),
WDL from the side to move, values in [-1, 1], unvisited children at Q = 0,
U = cpuct * P * sqrt(N_parent) / (1 + N). No root noise (this is analysis).
Leaves are evaluated in small batches with virtual loss.

Repetitions follow the self-play rules: a position occurring for the third time
(counting the game so far and the searched line, since the last capture or pawn
move) is an automatic draw, both inside the tree and as the game's end.

- mlh_eps: the self-play engine's moves-left tiebreak: among children,
  score += eps * (1 - 2 W) * tanh(plies / 25), W = the mover's win chance
  (approximated by its mean value), plies = the backed-up moves-left estimate.
  When winning it prefers shorter wins; when losing, longer games.
- twofold_draw (off by default): also score a position's second occurrence as a
  draw inside the tree, Leela-style.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import bulletchess
import chess
import numpy as np
import torch

from src.models.factory import model_from_checkpoint
from src.models.small.fen_encoder import fen_to_tensor
from src.models.small.vocab import policy_to_idx

HISTORY = 7


def policy_key(move: chess.Move) -> str:
    """The vocabulary's name for a move (knight promotions are the bare move)."""
    u = move.uci()
    return u[:4] if move.promotion == chess.KNIGHT else u


def encode(board: chess.Board) -> np.ndarray:
    """[8, 8, 111] input for the position, history taken from the board's move stack."""
    b = board.copy()
    current_rep = b.is_repetition(2)
    past_fens, past_reps = [], []
    for _ in range(HISTORY):
        if not b.move_stack:
            break
        b.pop()
        past_fens.append(b.fen())
        past_reps.append(b.is_repetition(2))
    past = [bulletchess.Board.from_fen(f) for f in past_fens]
    return fen_to_tensor(board.fen(), past, past_reps, current_rep)


class Net:
    """A checkpoint on the CPU. evaluate() returns legal-move priors and WDL per board."""

    def __init__(self, path: str | Path, threads: int = 4) -> None:
        torch.set_num_threads(threads)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model = model_from_checkpoint(ckpt, "cpu").eval()
        self.spec = dict(ckpt.get("model_spec") or {})
        self.params = sum(p.numel() for p in self.model.parameters())
        self.outcome_heads = (ckpt.get("config") or {}).get("search_heads", "search") == "outcome"
        self.step = int(ckpt.get("step", 0) or 0)

    @classmethod
    def wrap(cls, model: torch.nn.Module, outcome_heads: bool = True) -> Net:
        """A Net around an in-memory model (tests)."""
        net = cls.__new__(cls)
        net.model, net.spec, net.step = model.eval(), {}, 0
        net.params = sum(p.numel() for p in model.parameters())
        net.outcome_heads = outcome_heads
        return net

    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict[chess.Move, float], tuple[float, float, float], float]]:
        """Per board: legal-move priors, WDL for the side to move, estimated plies left."""
        x = torch.from_numpy(np.stack([encode(b) for b in boards]))
        p, v, vq, mlh, mlh_q = self.model(x, return_all=True)
        wdl = torch.softmax(v if self.outcome_heads else vq, dim=-1).numpy()
        plies = (mlh if self.outcome_heads else mlh_q).float().clamp_min(0).numpy() * 25.0
        logits = p.float().numpy()
        out = []
        for i, b in enumerate(boards):
            moves = list(b.legal_moves)
            idx = [policy_to_idx.get(policy_key(m), -1) for m in moves]
            lg = np.array([logits[i, j] if j >= 0 else -1e9 for j in idx], dtype=np.float64)
            pr = np.exp(lg - lg.max()) if len(lg) else lg
            pr = pr / pr.sum() if len(pr) else pr
            out.append(({m: float(q) for m, q in zip(moves, pr)}, tuple(float(t) for t in wdl[i]), float(plies[i])))
        return out


def terminal_value(board: chess.Board, twofold: bool = False) -> float | None:
    """Value for the side to move if the game is over (-1 mated, 0 drawn), else None.

    twofold: count a position seen before as a draw (search tree rule)."""
    if board.is_checkmate():
        return -1.0
    if (board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100
            or board.is_repetition(2 if twofold else 3)):
        return 0.0
    return None


def end_reason(board: chess.Board) -> str | None:
    """Why the game is over by the rules (an actual threefold, not a claimable one), else None."""
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.halfmove_clock >= 100:
        return "fifty-move rule"
    if board.is_repetition(3):
        return "threefold repetition"
    return None


@dataclass
class Node:
    board: chess.Board
    prior: float = 1.0
    n: int = 0
    w: float = 0.0  # sum of values from the point of view of the side that moved INTO this node
    vloss: int = 0
    children: dict[chess.Move, Node] = field(default_factory=dict)
    expanded: bool = False
    terminal: float | None = None
    wdl: tuple[float, float, float] | None = None
    plies_est: float = 0.0  # network's moves-left estimate at this node
    plies_sum: float = 0.0  # backed-up moves-left, summed over visits

    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


def search(net: Net, board: chess.Board, sims: int, cpuct: float = 0.65, batch: int = 8,
           twofold_draw: bool = False, mlh_eps: float = 0.05) -> dict:
    """PUCT search from `board`. Returns the root, its evaluation and timing."""
    t0 = time.time()
    root = Node(board.copy())
    root.terminal = terminal_value(root.board)
    if root.terminal is None:
        (priors, wdl, plies), = net.evaluate([root.board])
        _expand(root, priors, wdl, plies, twofold_draw)
    done = 0
    while done < sims and root.terminal is None:
        leaves, paths = [], []
        for _ in range(min(batch, sims - done)):
            path = [root]
            node = root
            while node.expanded and node.terminal is None and node.children:
                node = _select(node, cpuct, mlh_eps)
                path.append(node)
            for nd in path[1:]:
                nd.vloss += 1
            if node.terminal is None and not node.expanded and any(node is l for l in leaves):
                for nd in path[1:]:
                    nd.vloss -= 1
                break  # collision: evaluate what we have
            leaves.append(node)
            paths.append(path)
        pending = [l for l in leaves if l.terminal is None and not l.expanded]
        if pending:
            for l, (priors, wdl, plies) in zip(pending, net.evaluate([l.board for l in pending])):
                _expand(l, priors, wdl, plies, twofold_draw)
        for leaf, path in zip(leaves, paths):
            for nd in path[1:]:
                nd.vloss -= 1
            v = leaf.terminal if leaf.terminal is not None else (leaf.wdl[0] - leaf.wdl[2])
            plies = 0.0 if leaf.terminal is not None else leaf.plies_est
            # v is from the side to move at the leaf; the node's w is from the mover's side.
            for nd in reversed(path):
                v = -v
                nd.n += 1
                nd.w += v
                nd.plies_sum += plies
                plies += 1.0
        done += len(leaves)
    return {"root": root, "sims": done, "seconds": time.time() - t0}


def _expand(node: Node, priors: dict[chess.Move, float], wdl: tuple[float, float, float], plies: float,
            twofold_draw: bool) -> None:
    node.wdl, node.plies_est = wdl, plies
    for m, p in priors.items():
        child = node.board.copy(stack=True)
        child.push(m)
        c = Node(child, prior=p)
        c.terminal = terminal_value(child, twofold=twofold_draw)
        node.children[m] = c
    node.expanded = True


def _select(node: Node, cpuct: float, mlh_eps: float = 0.0) -> Node:
    sq = math.sqrt(max(1, node.n))
    best, best_s = None, -math.inf
    for c in node.children.values():
        n = c.n + c.vloss
        q = (c.w - c.vloss) / n if n else 0.0  # a pending visit counts as a loss for the mover
        s = q + cpuct * c.prior * sq / (1 + n)
        if mlh_eps and c.n:
            # (1 - 2 W) ~ -Q: winning prefers fewer plies left, losing prefers more.
            s -= mlh_eps * (c.w / c.n) * math.tanh(c.plies_sum / c.n / 25.0)
        if s > best_s:
            best, best_s = c, s
    return best


def analyse(net: Net, board: chess.Board, sims: int, **search_kw) -> dict:
    """Everything the board UI shows for a position."""
    res = search(net, board, sims, **search_kw)
    root = res["root"]
    rows = []
    for m, c in root.children.items():
        rows.append({
            "uci": m.uci(), "san": board.san(m), "prior": c.prior,
            "visits": c.n, "share": c.n / max(1, root.n) if sims else 0.0,
            "score": (c.q() + 1) / 2 if c.n else None,  # expected score for the side to move
        })
    rows.sort(key=lambda r: (r["visits"], r["prior"]), reverse=True)
    best = rows[0]["uci"] if rows else None
    return {"moves": rows, "best": best, "wdl": root.wdl, "sims": res["sims"],
            "seconds": round(res["seconds"], 3), "status": end_reason(board)}
