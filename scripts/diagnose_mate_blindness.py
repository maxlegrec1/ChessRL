"""Diagnose why step_9900 misses mate-in-1 in the brawl PGN.

For each black-to-move position in the game where mate-in-1 exists:
  - run step_9900 raw NN forward pass, extract policy prob on EVERY mate move,
    plus the model's argmax move, V[W], Vq[W], policy entropy, max-prior.
  - run a freshly random-initialized SmallModel for comparison.
  - report which would "find" the mate (argmax over legal mates a mate move?
    or: among legal moves, is a mate move in top-K policy?).

Then on one position, run MCTS-600 and dump visit distribution.
"""
from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import chess
import chess.pgn
import numpy as np
import torch

ROOT = Path("/mnt/2tb_2/ChessRL")
sys.path.insert(0, str(ROOT))

from src.models.small.fen_encoder import fen_to_tensor
from src.models.small.small_model import SmallModel
from src.models.small.vocab import policy_index, policy_to_idx


PGN = """[Event "MCTS brawl"]
[Site "local"]
[Date "2026.05.05"]
[Round "2"]
[White "step_00008500"]
[Black "step_00009900"]
[Result "1/2-1/2"]

1. e4 e6 2. d4 d5 3. e5 c5 4. Qh5 g6 5. Bg5 gxh5 6. Kd1 Qxg5 7. Nf3 Qg4 8. Rg1
Nh6 9. Rh1 Rg8 10. Ke1 Qf4 11. Bb5+ Kd8 12. Kd1 Rg4 13. Rg1 cxd4 14. Bf1 Bc5
15. Rh1 Rh4 16. Rg1 Qxf3+ 17. gxf3 b6 18. Rg7 Bf8 19. Rxf7 Nxf7 20. f4 Bh6 21.
Ke1 Bxf4 22. Kd1 Nxe5 23. Nd2 d3 24. Nf3 Nxf3 25. cxd3 Ke8 26. Rb1 Kd8 27. Ra1
e5 28. Rb1 Bh3 29. Rc1 Bg4 30. Rb1 Bh3 31. Ra1 Bg4 32. Rb1 a6 33. Ra1 Ke8 34.
Rb1 Kd8 35. Ra1 Kc8 36. Rb1 Rxh2 37. Rc1+ Kb7 38. Rc2 e4 39. Re2 exd3 40. Rd2
d4 41. Rxd3 h6 42. Kc2 Rxf2+ 43. Kb3 Rxf1 44. Re3 dxe3 45. a4 e2 46. Kc4 e1=Q
47. Kb3 Rf2 48. Ka3 Qe2 49. a5 Qc2 50. Ka2 Qd2 51. Ka3 Qc2 52. Ka2 Qe2 53. Ka3
Qxb2+ 54. Ka4 Qc2+ 55. Ka3 Qe2 56. Kb3 Qb2+ 57. Ka4 Rd2 58. axb6 Qc2+ 59. Ka5
Qc1 60. Ka4 Qc2+ 61. Ka5 Qb2 62. Ka4 Rd1 63. Ka5 Re1 64. Ka4 Qc2+ 65. Ka3 Qd2
66. Kb3 Re2 67. Ka3 Rh2 68. Kb3 Rf2 69. Ka3 Kxb6 70. Kb3 Qe2 71. Ka3 Qc2 72.
Kb4 Qc1 73. Ka4 Qf1 74. Ka3 Qc1+ 75. Kb3 Qc2+ 76. Ka3 Rd2 77. Kb4 Qc1 78. Ka4
Qd1+ 79. Ka3 Qf1 80. Kb3 Qe1 81. Ka3 Rc2 82. Kb3 Qe2 83. Ka3 Rd2 84. Kb3 Qf2
85. Ka3 Qf1 86. Kb3 Qd1+ 87. Ka3 Qe1 88. Kb3 Qe2 89. Ka3 Qf2 90. Kb3 Qf1 91.
Ka3 Qe1 92. Kb3 Rc2 93. Ka3 Rc1 94. Kb3 Rc2 95. Ka3 Nd2 96. Ka4 Rb2 97. Ka3 Qe2
98. Ka4 Qf2 99. Ka3 Qg2 100. Ka4 Qg1 101. Ka3 Qd1 102. Kxb2 Qf1 103. Ka3 Qf2
104. Ka4 Qe2 105. Kb4 Qe1 106. Ka4 Kc5 107. Ka3 Kd4 108. Kb4 Ke3 109. Ka4 Qe2
110. Kb4 Qf2 111. Ka4 Qg2 112. Kb4 Qh2 113. Ka4 a5 114. Kb5 Qf2 115. Kc5 Qf1
116. Kd5 Qd1 117. Kc5 Qh1 118. Kb5 Qe1 119. Kc5 Qh1 120. Kb5 Qh2 121. Kc5 a4
122. Kd5 Bf3+ 123. Kc5 Qh1 124. Kb5 Bg2 125. Kc5 a3 126. Kb6 a2 127. Kb5 Qh2
128. Kc5 a1=Q 129. Kb5 Qf1+ 130. Kc5 Qd1 131. Kb5 Qb1+ 132. Kc5 Qd1 133. Kb5
Qf1+ 134. Kc5 Qe1 135. Kb5 Qhh1 136. Kc5 Qeg1 137. Kb4 Bf1 138. Kc3 Qgg2 139.
Kb2 Qgh2 140. Kc3 Q2g1 141. Kb2 Qgg2 142. Kc1 Qgh2 143. Kd1 Q1g2 144. Kc1 Qgg1
145. Kd1 Qhg2 146. Kc1 Q2h2 147. Kd1 Qhf2 148. Kc1 Qe2 149. Kb2 Qgg2 150. Kc1
Qh2 151. Kb2 Qeg2 152. Kc1 Nc6 153. Kd1 Rb8 154. Kc2 Nd4+ 155. Kc1 N4f3 156.
Kc2 Rb2+ 157. Kc3 Qe2 158. Kxb2 Qeg2 159. Ka1 Qgg1 160. Ka2 Qhf2 161. Kb2 Qgg2
162. Ka1 Qh2 163. Ka2 Qfg2 164. Kb2 Qhh3 165. Ka1 Qgh2 166. Ka2 Q2g2 167. Kb2
Qgh2 168. Ka1 Q2g2 169. Ka2 Qgh2 170. Kb2 Q2g2 171. Ka2 Qhg3 172. Kb2 Q2h2 173.
Ka2 Qgg2 174. Kb2 Qhh1 175. Ka1 Qhg1 176. Ka2 Q2h2 177. Kb2 Qhf2 178. Ka2 Qe2
179. Kb2 Qeg2 180. Ka1 Qe2 181. Ka2 Kf2 182. Kb2 Qe1 183. Ka3 Qa1+ 184. Kb4 Qe1
185. Ka4 Qd1+ 186. Kb4 Qa1 187. Kc5 Qb1 188. Kd5 Qc1 189. Ke6 Qd1 190. Kd5 Qb1
191. Kc5 Qb2 192. Kd5 Qc2 193. Ke6 Qc1 194. Kf6 Qd1 195. Ke7 Qb1 196. Ke6 Qc1
197. Kd5 Qg2 198. Ke6 Qb1 199. Kd5 Qb2 200. Kc5 Qh2 201. Kd5 Ne4 202. Ke6 Qb1
203. Kd7 Qg2 204. Kc6 Qg1 205. Kd5 Qc1 206. Ke6 Qc2 207. Kd5 Qb2 208. Kc6
1/2-1/2"""


def find_mate_moves(board: chess.Board) -> List[chess.Move]:
    mates = []
    for mv in board.legal_moves:
        board.push(mv)
        if board.is_checkmate():
            mates.append(mv)
        board.pop()
    return mates


def collect_black_mate_positions() -> List[Tuple[int, str, List[str]]]:
    """Returns list of (ply_number, fen_before_black_move, [mate_uci ...])."""
    game = chess.pgn.read_game(io.StringIO(PGN))
    out = []
    board = game.board()
    ply = 0
    for mv in game.mainline_moves():
        ply += 1
        # before black plays, position is black-to-move
        if board.turn == chess.BLACK:
            mates = find_mate_moves(board)
            if mates:
                out.append((ply, board.fen(), [m.uci() for m in mates]))
        board.push(mv)
    return out


@torch.no_grad()
def evaluate_model(model: SmallModel, fen: str, device: str = "cuda"):
    model.eval()
    board = chess.Board(fen)
    arr = fen_to_tensor(fen)
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    x = arr.unsqueeze(0).float().to(device)
    p_logits, v_logits, vq_logits, _mlh = model(x, return_all=True)
    p = torch.softmax(p_logits[0], dim=-1).cpu().numpy()
    v = torch.softmax(v_logits[0], dim=-1).cpu().numpy()
    vq = torch.softmax(vq_logits[0], dim=-1).cpu().numpy()

    # legal-move masking
    legal_ucis = [m.uci() for m in board.legal_moves]
    legal_idx = [policy_to_idx[u] for u in legal_ucis if u in policy_to_idx]
    legal_p = np.zeros_like(p)
    for u in legal_ucis:
        if u in policy_to_idx:
            legal_p[policy_to_idx[u]] = p[policy_to_idx[u]]
    s = legal_p.sum()
    if s > 0:
        legal_p = legal_p / s
    return {
        "n_legal": len(legal_ucis),
        "legal_p": legal_p,
        "v": v,  # [L, D, W] from side-to-move POV
        "vq": vq,
        "argmax_uci": policy_index[int(legal_p.argmax())] if s > 0 else None,
        "max_prior": float(legal_p.max()),
        "entropy": float(-(legal_p[legal_p > 0] * np.log(legal_p[legal_p > 0])).sum()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = ROOT / "artifacts/runs/20260504_014403_e717/checkpoints/step_00009900.pt"
    print(f"Loading {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("model", state.get("state_dict", state))
    m9900 = SmallModel().to(device)
    m9900.load_state_dict(sd)

    torch.manual_seed(0)
    mrand = SmallModel().to(device)

    positions = collect_black_mate_positions()
    print(f"\nFound {len(positions)} black-to-move positions with mate-in-1.\n")
    print(f"{'ply':>4} {'#mates':>6} {'#legal':>6} | {'9900_pmate':>10} {'9900_pmax':>9} {'9900_argmax':>12} {'9900_V[W]':>9} {'9900_Vq[W]':>10} {'9900_H':>7} | {'rand_pmate':>10} {'rand_pmax':>9} {'rand_H':>7} {'mate_in_topK?':>14}")

    n_9900_finds = 0
    n_rand_finds = 0
    n_9900_topk = {1: 0, 5: 0, 20: 0}
    n_rand_topk = {1: 0, 5: 0, 20: 0}
    sum_9900_pmate = 0.0
    sum_rand_pmate = 0.0

    for ply, fen, mates in positions:
        e9 = evaluate_model(m9900, fen, device)
        er = evaluate_model(mrand, fen, device)
        mate_idx = [policy_to_idx[u] for u in mates if u in policy_to_idx]
        p9_mate = max((e9["legal_p"][i] for i in mate_idx), default=0.0)
        pr_mate = max((er["legal_p"][i] for i in mate_idx), default=0.0)
        sum_9900_pmate += p9_mate
        sum_rand_pmate += pr_mate

        # rank of best mate move in legal-policy distribution
        order9 = np.argsort(-e9["legal_p"])
        orderr = np.argsort(-er["legal_p"])
        rank9 = min((int(np.where(order9 == i)[0][0]) for i in mate_idx), default=999)
        rankr = min((int(np.where(orderr == i)[0][0]) for i in mate_idx), default=999)
        for k in (1, 5, 20):
            if rank9 < k:
                n_9900_topk[k] += 1
            if rankr < k:
                n_rand_topk[k] += 1
        if e9["argmax_uci"] in mates:
            n_9900_finds += 1
        if er["argmax_uci"] in mates:
            n_rand_finds += 1

        in_top = f"9900:{rank9} rnd:{rankr}"
        print(
            f"{ply:>4} {len(mates):>6} {e9['n_legal']:>6} | "
            f"{p9_mate:>10.4f} {e9['max_prior']:>9.4f} {str(e9['argmax_uci']):>12} "
            f"{e9['v'][2]:>9.3f} {e9['vq'][2]:>10.3f} {e9['entropy']:>7.3f} | "
            f"{pr_mate:>10.4f} {er['max_prior']:>9.4f} {er['entropy']:>7.3f} {in_top:>14}"
        )

    n = max(1, len(positions))
    print()
    print(f"=== Summary across {n} mate-in-1 positions (black to move) ===")
    print(f"step_9900 argmax-is-mate:  {n_9900_finds}/{n}  ({100*n_9900_finds/n:.1f}%)")
    print(f"random    argmax-is-mate:  {n_rand_finds}/{n}  ({100*n_rand_finds/n:.1f}%)")
    print(f"step_9900 mean p(mate):    {sum_9900_pmate/n:.4f}")
    print(f"random    mean p(mate):    {sum_rand_pmate/n:.4f}")
    for k in (1, 5, 20):
        print(f"  top-{k}: 9900={n_9900_topk[k]}/{n}  random={n_rand_topk[k]}/{n}")


if __name__ == "__main__":
    main()
