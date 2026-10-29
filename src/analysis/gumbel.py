"""Gumbel AlphaZero root search on the CPU analysis engine's tree (src/analysis/engine.py).

Danihelka, Guez, Schrittwieser and Silver, "Policy improvement by planning with Gumbel"
(ICLR 2022); details and defaults follow the reference implementation
(google-deepmind/mctx: gumbel_muzero_policy, qtransform_completed_by_mix_value, seq_halving).

At the root, the moves to consider are sampled without replacement: the `considered` best
by logit + Gumbel noise (the Gumbel-top-k trick samples k distinct moves in proportion to
the policy). The simulations are split between them by sequential halving: each candidate
gets the same number of simulations, then the better half (by noise + logit + sigma(Q))
gets the next share, and so on until one move is left; that move is played. Below the root,
"gumbel" picks the child whose visit share lags the improved policy the most
(deterministic); "puct" uses the engine's PUCT rule instead.

The training target is the improved policy softmax(logit + sigma(completed Q)) over every
legal move. Completed Q is the searched Q for visited moves and, for the others, a mix of
the network's value at the node and the visited moves' Q. sigma rescales the completed Q to
[0, 1] over the node's moves and multiplies by (c_visit + the most visits) * c_scale.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import chess
import numpy as np

from src.analysis.engine import Net, Node, _expand, _select, terminal_value


def considered_visit_sequence(considered: int, sims: int) -> list[int]:
    """Per simulation, the visit count a root move must have to be simulated next.

    The first `considered` entries are 0 (each candidate gets a first visit), then each
    halving round gives the remaining candidates equal extra visits (mctx's
    get_sequence_of_considered_visits)."""
    if considered <= 1:
        return list(range(sims))
    log2 = math.ceil(math.log2(considered))
    seq: list[int] = []
    visits = [0] * considered
    k = considered
    while len(seq) < sims:
        extra = max(1, int(sims / (log2 * k)))
        for _ in range(extra):
            seq.extend(visits[:k])
            for i in range(k):
                visits[i] += 1
        k = max(2, k // 2)
    return seq[:sims]


def completed_q(
    node: Node, moves: list[chess.Move], priors: np.ndarray, c_visit: float, c_scale: float
) -> np.ndarray:
    """sigma(completed Q) of the node's moves, from the side to move at the node."""
    n = np.array([node.children[m].n for m in moves], dtype=np.float64)
    q = np.array([node.children[m].q() for m in moves], dtype=np.float64)
    raw = node.wdl[0] - node.wdl[2]
    visited = n > 0
    if visited.any():
        p = np.maximum(priors, np.finfo(np.float64).tiny)
        mean_q = float((p[visited] * q[visited]).sum() / p[visited].sum())
        mixed = (raw + n.sum() * mean_q) / (n.sum() + 1.0)
    else:
        mixed = raw
    cq = np.where(visited, q, mixed)
    lo, hi = cq.min(), cq.max()
    cq = (cq - lo) / max(hi - lo, 1e-8)
    return (c_visit + n.max()) * c_scale * cq


def _logits(priors: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(priors, 1e-30))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def gumbel_search(
    net: Net,
    board: chess.Board,
    sims: int,
    *,
    considered: int = 16,
    rng: np.random.Generator | None = None,
    interior: str = "gumbel",
    cpuct: float = 0.65,
    mlh_eps: float = 0.05,
    c_visit: float = 50.0,
    c_scale: float = 0.1,
    twofold_draw: bool = False,
) -> dict:
    """Search `board` with `sims` simulations. Returns the root, the move to play, the
    improved policy (the training target) and the number of network evaluations."""
    rng = rng if rng is not None else np.random.default_rng()
    root = Node(board.copy())
    root.terminal = terminal_value(root.board)
    if root.terminal is not None:
        return {"root": root, "action": None, "improved": {}, "evals": 0}
    ((priors, wdl, plies),) = net.evaluate([root.board])
    _expand(root, priors, wdl, plies, twofold_draw)
    evals = 1

    def select_gumbel(node: Node) -> Node:
        moves = list(node.children)
        p = np.array([node.children[m].prior for m in moves])
        pi = _softmax(_logits(p) + completed_q(node, moves, p, c_visit, c_scale))
        n = np.array([node.children[m].n for m in moves], dtype=np.float64)
        return node.children[moves[int(np.argmax(pi - n / (1.0 + n.sum())))]]

    def select_puct(node: Node) -> Node:
        return _select(node, cpuct, mlh_eps)

    select: Callable[[Node], Node] = select_gumbel if interior == "gumbel" else select_puct

    moves = list(root.children)
    prior = np.array([root.children[m].prior for m in moves])
    logits = _logits(prior)
    g = rng.gumbel(size=len(moves))
    for target in considered_visit_sequence(min(considered, len(moves)), sims):
        n = np.array([root.children[m].n for m in moves])
        score = g + logits + completed_q(root, moves, prior, c_visit, c_scale)
        a = int(np.argmax(np.where(n == target, score, -np.inf)))
        evals += _simulate(net, [root, root.children[moves[a]]], select, twofold_draw)

    n = np.array([root.children[m].n for m in moves])
    cq = completed_q(root, moves, prior, c_visit, c_scale)
    action = moves[int(np.argmax(np.where(n == n.max(), g + logits + cq, -np.inf)))]
    improved = _softmax(logits + cq)
    return {
        "root": root,
        "action": action,
        "improved": {m: float(x) for m, x in zip(moves, improved, strict=True)},
        "evals": evals,
    }


def _simulate(
    net: Net, path: list[Node], select: Callable[[Node], Node], twofold_draw: bool
) -> int:
    """One simulation below path[-1]; backs its value up through path. Returns evaluations."""
    node = path[-1]
    while node.expanded and node.terminal is None and node.children:
        node = select(node)
        path.append(node)
    evals = 0
    if node.terminal is None and not node.expanded:
        ((priors, wdl, plies),) = net.evaluate([node.board])
        _expand(node, priors, wdl, plies, twofold_draw)
        evals = 1
    v = node.terminal if node.terminal is not None else node.wdl[0] - node.wdl[2]
    left = 0.0 if node.terminal is not None else node.plies_est
    # v is from the side to move at the leaf; a node's w is from the side that moved into it.
    for nd in reversed(path):
        v = -v
        nd.n += 1
        nd.w += v
        nd.plies_sum += left
        left += 1.0
    return evals
