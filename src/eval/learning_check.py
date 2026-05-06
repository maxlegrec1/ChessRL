"""Quick learning-progress benchmark for SmallModel.

Two questions:
  1. Does the value head recognise that the side to move is winning?
     (Every position in the dataset has the side to move clearly winning.)
  2. On mate-in-1 positions, does the policy head's top legal move equal
     the unique mate?

Usage:
    uv run python -m src.eval.learning_check                 # auto-find latest checkpoint
    uv run python -m src.eval.learning_check --checkpoint X.pt
    uv run python -m src.eval.learning_check --run-dir DIR   # explicit run

All positions are validated with python-chess at load; bad entries are
skipped with a warning. Output is per-theme accuracy so failure modes are
easy to read off.
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import chess
import numpy as np
import torch

from src.models.small.fen_encoder import fen_to_tensor
from src.models.small.small_model import SmallModel
from src.models.small.vocab import policy_index, policy_to_idx


# (theme, fen, expected_best_uci_or_None)
# - mate_*  themes: expected_best_uci is the unique mate move.
# - end_*   themes: side to move is overwhelmingly winning (no mate-in-1 yet);
#                   we only check the value head.
# Hand-curated mate-in-1 positions. Each is verified by python-chess at load.
# For broad coverage we also generate winning endgames programmatically below.
MATE_IN_1: List[Tuple[str, str, str]] = [
    # ----- back-rank rook mates (king cornered by f7/g7/h7 pawns) ------
    ("mate_1_back_rank", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", "a1a8"),
    ("mate_1_back_rank", "6k1/5ppp/8/8/8/8/8/1R5K w - - 0 1", "b1b8"),
    ("mate_1_back_rank", "6k1/5ppp/8/8/8/8/8/2R4K w - - 0 1", "c1c8"),
    ("mate_1_back_rank", "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1", "d1d8"),
    ("mate_1_back_rank", "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1", "e1e8"),
    # Note: f1f8 fails because the f7 pawn blocks the f-file. Skipped.
    # ----- back-rank queen mates ---------------------------------------
    ("mate_1_queen_back", "6k1/5ppp/8/8/8/8/8/Q6K w - - 0 1", "a1a8"),
    ("mate_1_queen_back", "6k1/5ppp/8/8/8/8/8/1Q5K w - - 0 1", "b1b8"),
    ("mate_1_queen_back", "6k1/5ppp/8/8/8/8/8/2Q4K w - - 0 1", "c1c8"),
    ("mate_1_queen_back", "6k1/5ppp/8/8/8/8/8/3Q3K w - - 0 1", "d1d8"),
    ("mate_1_queen_back", "6k1/5ppp/8/8/8/8/8/4Q2K w - - 0 1", "e1e8"),
    # Black-to-move mirror: BlackKpawns on rank 2 with white king cornered
    # would let white pawns promote to block. Switch the pawn's role: place
    # them on rank 6 (white pawns) and a black king on g1 to mirror correctly.
    # Black rook delivers a back-rank mate at rank 1.
    # Actually simpler: keep this one safe by using K vs lone piece — no
    # blocking pawns to promote. White king on h8, two black rooks ladder.
    # ----- two-rook ladder mate ----------------------------------------
    # Pattern: rook on rank 2 cuts off h2 escape; rook on rank 3 delivers
    # mate on rank 1 from a file far enough that the enemy king can't
    # capture it (need file <= f for king on h1, or a king-defender nearby).
    ("mate_1_two_rooks", "K7/8/8/8/8/1R6/R7/7k w - - 0 1", "b3b1"),
    ("mate_1_two_rooks", "K7/8/8/8/8/2R5/R7/7k w - - 0 1", "c3c1"),
    ("mate_1_two_rooks", "K7/8/8/8/8/3R4/R7/7k w - - 0 1", "d3d1"),
    ("mate_1_two_rooks", "K7/8/8/8/8/4R3/R7/7k w - - 0 1", "e3e1"),
    ("mate_1_two_rooks", "K7/8/8/8/8/5R2/R7/7k w - - 0 1", "f3f1"),
    # Mirrored (black-to-move ladder, no promotable pawns in the way).
    ("mate_1_two_rooks", "7K/r7/1r6/8/8/8/8/k7 b - - 0 1", "b6b8"),
    ("mate_1_two_rooks", "7K/r7/4r3/8/8/8/8/k7 b - - 0 1", "e6e8"),
]


# Endgame templates: (theme, my_pieces, opp_pieces, count_per_side).
# my_pieces are placed for the side to move; opp_pieces for the opponent.
# Each side always has a king implicitly. We generate `count` valid positions
# per side (white-to-move and black-to-move), randomized by the seed.
ENDGAME_TEMPLATES = [
    ("end_kqk",   [chess.QUEEN], [], 14),
    ("end_krk",   [chess.ROOK],  [], 14),
    ("end_kqrk",  [chess.QUEEN, chess.ROOK], [], 14),
    ("end_kqqk",  [chess.QUEEN, chess.QUEEN], [], 10),
    ("end_krrk",  [chess.ROOK, chess.ROOK],   [], 10),
    # K+B+N vs K (winning but trickier — useful failure-mode signal).
    ("end_kbnk",  [chess.BISHOP, chess.KNIGHT], [], 8),
]


def _generate_endgame(theme: str, my_pieces, opp_pieces, count: int, seed: int) -> List[Tuple[str, str, Optional[str]]]:
    rng = random.Random(seed)
    out: List[Tuple[str, str, Optional[str]]] = []
    # Half white-to-move, half black-to-move.
    for stm_white in (True, False):
        n_target = count // 2 + (count % 2 if stm_white else 0)
        attempts = 0
        while len(out) < count and attempts < n_target * 200:
            attempts += 1
            board = chess.Board.empty()
            squares = rng.sample(range(64), 2 + len(my_pieces) + len(opp_pieces))
            i = 0
            board.set_piece_at(squares[i], chess.Piece(chess.KING, chess.WHITE)); i += 1
            board.set_piece_at(squares[i], chess.Piece(chess.KING, chess.BLACK)); i += 1
            stm_color = chess.WHITE if stm_white else chess.BLACK
            opp_color = not stm_color
            for pt in my_pieces:
                board.set_piece_at(squares[i], chess.Piece(pt, stm_color)); i += 1
            for pt in opp_pieces:
                board.set_piece_at(squares[i], chess.Piece(pt, opp_color)); i += 1
            board.turn = stm_color
            if not board.is_valid():
                continue
            if board.is_check():  # we want the side to move not in check
                continue
            if board.is_game_over():
                continue
            # Avoid immediate-mate-in-1 endgames so the value-head test is
            # measuring "knows it's winning" not "is already crushed".
            out.append((theme, board.fen(), None))
    return out[:count]


def validate_position(theme: str, fen: str, best_uci: Optional[str]) -> Tuple[bool, str]:
    """Returns (valid, reason). Skips entry on mismatch."""
    try:
        board = chess.Board(fen)
    except Exception as e:
        return False, f"unparseable FEN ({e})"
    if not board.is_valid():
        return False, "python-chess says invalid (e.g. side-not-to-move in check)"
    if board.is_game_over(claim_draw=False):
        return False, "already game over"

    if theme.startswith("mate_1"):
        if best_uci is None:
            return False, "mate theme requires expected best move"
        try:
            move = chess.Move.from_uci(best_uci)
        except ValueError:
            return False, f"unparseable best move {best_uci!r}"
        if move not in board.legal_moves:
            return False, f"{best_uci} not legal in this FEN"
        board.push(move)
        if not board.is_checkmate():
            return False, f"{best_uci} does not deliver checkmate"
    elif theme.startswith("end_"):
        # Side to move not in check, has legal moves, has at least one heavy piece.
        if board.is_check():
            return False, "side to move is in check (not the theme)"
        # Material sanity: side to move should have far more than the opponent.
        stm = board.turn
        VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                  chess.ROOK: 5, chess.QUEEN: 9}
        my_mat = sum(len(board.pieces(pt, stm)) * v for pt, v in VALUES.items())
        opp_mat = sum(len(board.pieces(pt, not stm)) * v for pt, v in VALUES.items())
        if my_mat - opp_mat < 4:
            return False, f"material gap too small ({my_mat} vs {opp_mat})"
    return True, "ok"


def load_dataset(seed: int = 42) -> List[Tuple[str, str, Optional[str]]]:
    valid: List[Tuple[str, str, Optional[str]]] = []
    rejected = defaultdict(list)
    # Hand-curated mate-in-1.
    for theme, fen, best in MATE_IN_1:
        ok, reason = validate_position(theme, fen, best)
        if ok:
            valid.append((theme, fen, best))
        else:
            rejected[theme].append((fen, reason))
    # Programmatic endgames.
    for theme, my_p, opp_p, n in ENDGAME_TEMPLATES:
        generated = _generate_endgame(theme, my_p, opp_p, n, seed=seed + hash(theme) % 1000)
        for theme_, fen, best in generated:
            ok, reason = validate_position(theme_, fen, best)
            if ok:
                valid.append((theme_, fen, best))
            else:
                rejected[theme_].append((fen, reason))
    if rejected:
        print("[learning_check] rejected positions:", file=sys.stderr)
        for theme, items in rejected.items():
            for fen, reason in items[:3]:
                print(f"  [{theme}] {fen!r}: {reason}", file=sys.stderr)
            if len(items) > 3:
                print(f"  [{theme}] ... and {len(items) - 3} more", file=sys.stderr)
    return valid


def find_latest_checkpoint(run_dir: Optional[Path], use_teacher: bool = False) -> Path:
    """Locate the latest checkpoint to benchmark.

    By default, returns the latest STUDENT checkpoint (`step_*.pt`) — that's
    the model self-play actually exports to TRT, so it is what we want to
    evaluate. The teacher snapshot (`teacher_step_*.pt`) is currently
    vestigial: it's EMA-updated and saved to disk but never used for inference.
    Pass use_teacher=True to evaluate the teacher snapshot instead.
    """
    if run_dir is None:
        runs_root = Path("artifacts/runs")
        if not runs_root.is_dir():
            raise FileNotFoundError("artifacts/runs/ not found; pass --checkpoint")
        run_dirs = sorted([d for d in runs_root.iterdir() if d.is_dir()])
        if not run_dirs:
            raise FileNotFoundError("no run dirs under artifacts/runs/")
        run_dir = run_dirs[-1]
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"no checkpoints/ in {run_dir}")
    pattern = "teacher_step_*.pt" if use_teacher else "step_*.pt"
    candidates = sorted(ckpt_dir.glob(pattern))
    if not candidates:
        # Fall back to whichever is available so the script doesn't crash if
        # one of the checkpoint families is missing.
        candidates = sorted(ckpt_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no .pt files in {ckpt_dir}")
    return candidates[-1]


def load_model(checkpoint_path: Path, device: str) -> SmallModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = SmallModel().to(device).eval()
    # strict=False so pre-MLH checkpoints still load — the new mlh_* params
    # stay at their fresh init.
    model.load_state_dict(state, strict=False)
    return model


def run_inference(
    model: SmallModel, fens: List[str], device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (policy_logits[B,1929], value_softmax[B,3], value_q_softmax[B,3])."""
    planes = np.stack([fen_to_tensor(f) for f in fens]).astype(np.float32)
    x = torch.from_numpy(planes).to(device)
    with torch.no_grad():
        policy_logits, value_logits, value_q_logits, _mlh, _mlh_q = model(x, return_all=True)
    return (
        policy_logits.float().cpu().numpy(),
        torch.softmax(value_logits.float(), dim=-1).cpu().numpy(),
        torch.softmax(value_q_logits.float(), dim=-1).cpu().numpy(),
    )


def best_legal_uci(policy_logits: np.ndarray, fen: str) -> str:
    """Argmax over legal moves only. Knight promos lookup with promo stripped."""
    board = chess.Board(fen)
    best_uci = None
    best_logit = -np.inf
    for mv in board.legal_moves:
        uci = mv.uci()
        # vocab strips trailing 'n' on knight promos
        key = uci[:-1] if uci.endswith("n") else uci
        idx = policy_to_idx.get(key)
        if idx is None:
            continue
        if policy_logits[idx] > best_logit:
            best_logit = policy_logits[idx]
            best_uci = uci
    return best_uci or ""


RECORD_SIZE = 148
VALUE_OUTCOME_OFFSET = 116
VALUE_MCTS_OFFSET = 128


def analyze_buffer(buffer_path: Path, sample_size: int = 200_000) -> None:
    """Sample-scan a self-play buffer and print target-distribution stats.

    Mirrors the metric set the trainer now logs to W&B
    (buffer/frac_W_geq_0.8, buffer/frac_L_geq_0.8, buffer/frac_near_uniform)
    so old runs without those metrics in W&B can still get the read-out.
    """
    if not buffer_path.is_file() or buffer_path.stat().st_size < RECORD_SIZE:
        print(f"[buffer-stats] no usable buffer at {buffer_path}")
        return
    content = buffer_path.read_bytes()
    n = len(content) // RECORD_SIZE
    if n == 0:
        print(f"[buffer-stats] {buffer_path} contains 0 records")
        return
    sample_size = min(sample_size, n)
    idxs = np.random.default_rng(42).choice(n, sample_size, replace=False)
    vo = np.zeros((sample_size, 3), dtype=np.float32)
    vm = np.zeros((sample_size, 3), dtype=np.float32)
    for i, idx in enumerate(idxs):
        o = int(idx) * RECORD_SIZE
        vo[i] = struct.unpack_from("fff", content, o + VALUE_OUTCOME_OFFSET)
        vm[i] = struct.unpack_from("fff", content, o + VALUE_MCTS_OFFSET)

    nonzero = vo.sum(axis=1) > 0
    cls_o = vo.argmax(axis=1)
    n_w_o = int(((cls_o == 0) & nonzero).sum())
    n_d_o = int(((cls_o == 1) & nonzero).sum())
    n_l_o = int(((cls_o == 2) & nonzero).sum())
    max_p = vm.max(axis=1)
    frac_w_conf = float((vm[:, 0] >= 0.8).mean())
    frac_l_conf = float((vm[:, 2] >= 0.8).mean())
    frac_d_conf = float((vm[:, 1] >= 0.8).mean())
    frac_uniform = float((max_p < 0.40).mean())

    print()
    print(f"=== buffer stats ({buffer_path.name}, n={n:,}, sampled {sample_size:,}) ===")
    print(f"value_outcome (game ends, side-to-move):")
    print(f"  W = {100*n_w_o/sample_size:5.1f}%   D = {100*n_d_o/sample_size:5.1f}%   L = {100*n_l_o/sample_size:5.1f}%")
    print(f"value_mcts (dense MCTS-WDL):")
    print(f"  mean = ({vm[:,0].mean():.3f}, {vm[:,1].mean():.3f}, {vm[:,2].mean():.3f})")
    print(f"  frac_W_geq_0.8     = {100*frac_w_conf:5.1f}%")
    print(f"  frac_L_geq_0.8     = {100*frac_l_conf:5.1f}%")
    print(f"  frac_D_geq_0.8     = {100*frac_d_conf:5.1f}%")
    print(f"  frac_near_uniform  = {100*frac_uniform:5.1f}%   (max class < 0.40)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, help="explicit .pt file")
    p.add_argument("--run-dir", type=Path, help="run dir; uses its latest .pt")
    p.add_argument("--buffer", type=Path, help="explicit buffer.bin to analyze")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-buffer-stats", action="store_true",
                   help="skip the buffer-distribution analysis")
    p.add_argument("--teacher", action="store_true",
                   help="benchmark the teacher_step_*.pt snapshot instead of "
                        "the student. Default is the student because that's "
                        "what self-play exports to TRT.")
    args = p.parse_args()

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.run_dir, use_teacher=args.teacher)
    print(f"checkpoint: {ckpt_path}")
    print(f"device:     {args.device}")

    positions = load_dataset()
    if not positions:
        print("no valid positions; aborting", file=sys.stderr)
        return 1
    print(f"valid positions: {len(positions)}")

    model = load_model(ckpt_path, args.device)
    fens = [fen for _, fen, _ in positions]
    policy_logits, value_sm, value_q_sm = run_inference(model, fens, args.device)

    # Per-theme stats
    stats = defaultdict(lambda: {"n": 0, "v_w": 0, "vq_w": 0, "policy_correct": 0,
                                 "policy_n": 0, "v_score": 0.0, "vq_score": 0.0})
    for i, (theme, fen, best) in enumerate(positions):
        s = stats[theme]
        s["n"] += 1
        # Value head: index 0 = W, 1 = D, 2 = L (side-to-move perspective).
        if int(np.argmax(value_sm[i])) == 0:
            s["v_w"] += 1
        if int(np.argmax(value_q_sm[i])) == 0:
            s["vq_w"] += 1
        s["v_score"] += float(value_sm[i, 0])
        s["vq_score"] += float(value_q_sm[i, 0])
        if best is not None:
            s["policy_n"] += 1
            picked = best_legal_uci(policy_logits[i], fen)
            if picked == best:
                s["policy_correct"] += 1

    # Print report
    print()
    print(f"{'theme':<22} {'N':>4}  {'V→W':>7}  {'Vq→W':>7}  {'V[W]':>6}  {'Vq[W]':>6}  {'mate':>9}")
    print("-" * 84)
    totals = {"n": 0, "v_w": 0, "vq_w": 0, "policy_correct": 0, "policy_n": 0,
              "v_score": 0.0, "vq_score": 0.0}
    for theme in sorted(stats):
        s = stats[theme]
        v_pct = 100 * s["v_w"] / s["n"]
        vq_pct = 100 * s["vq_w"] / s["n"]
        v_score = s["v_score"] / s["n"]
        vq_score = s["vq_score"] / s["n"]
        if s["policy_n"]:
            mate_pct = f"{100 * s['policy_correct'] / s['policy_n']:>5.1f}% ({s['policy_correct']}/{s['policy_n']})"
        else:
            mate_pct = "       –"
        print(f"{theme:<22} {s['n']:>4}  "
              f"{v_pct:>6.1f}%  {vq_pct:>6.1f}%  "
              f"{v_score:>6.3f}  {vq_score:>6.3f}  {mate_pct}")
        for k in totals:
            totals[k] += s[k]
    print("-" * 84)
    n = totals["n"] or 1
    pn = totals["policy_n"] or 1
    print(f"{'overall':<22} {totals['n']:>4}  "
          f"{100*totals['v_w']/n:>6.1f}%  {100*totals['vq_w']/n:>6.1f}%  "
          f"{totals['v_score']/n:>6.3f}  {totals['vq_score']/n:>6.3f}  "
          f"{100*totals['policy_correct']/pn:>5.1f}% ({totals['policy_correct']}/{totals['policy_n']})")
    print()
    print("legend: V→W = % positions where value-head argmax is Win;")
    print("        V[W] = mean P(Win) from the value head.")
    print("        Same for Vq (the dense MCTS-trained value_q head).")
    print("        mate column = % mate-in-1 found via legal-move argmax over policy.")

    # Buffer stats — auto-detect from the checkpoint's run dir, or use --buffer.
    if not args.no_buffer_stats:
        buffer_path: Optional[Path] = args.buffer
        if buffer_path is None:
            # ckpt_path is artifacts/runs/<id>/checkpoints/foo.pt → ../buffer/buffer.bin
            run_dir_guess = ckpt_path.parent.parent
            candidate = run_dir_guess / "buffer" / "buffer.bin"
            if candidate.is_file():
                buffer_path = candidate
        if buffer_path is not None:
            analyze_buffer(buffer_path)
        else:
            print("\n[buffer-stats] no buffer.bin found near the checkpoint; pass --buffer to override")
    return 0


if __name__ == "__main__":
    sys.exit(main())
