"""Run MCTS-600 on every ply of a PGN game and report per-position evaluation.

For each ply (the position BEFORE that move was played), run MCTS-600 with the
latest engine and report:
  - ply, side-to-move, the move the model would pick, the move actually played
  - MCTS root WDL (visit-weighted-backed-up triple from search) — this is what
    the model THINKS the position is worth after 600 sims of search.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import chess
import chess.pgn

ROOT = Path("/mnt/2tb_2/ChessRL")
sys.path.insert(0, str(ROOT))

import _inference_cpp


PGN = """[Event "MCTS brawl"]
[Site "local"]
[Date "2026.05.06"]
[Round "6"]
[White "step_00010000"]
[Black "step_00015400"]
[Result "1/2-1/2"]

1. e4 c6 2. Nc3 d5 3. Nf3 dxe4 4. Ng5 Qd4 5. Qh5 g6 6. Ncxe4 gxh5 7. f4 Bf5 8.
Rg1 Bxe4 9. d3 Bh6 10. Ne6 Bxf4 11. Bxf4 Qxb2 12. Bc1 Qxa1 13. dxe4 Qxc1+ 14.
Ke2 Nf6 15. g4 Qxc2+ 16. Kf3 Nxe4 17. Rg3 Rg8 18. g5 Rxg5 19. Rxg5 fxe6 20. Ke3
Kd7 21. Bd3 Qb2 22. Bxe4 c5 23. Bd3 Nc6 24. Kf3 Nd4+ 25. Kg3 Qd2 26. Rg7 Kc6
27. Rf7 Rf8 28. Rxf8 c4 29. Rf7 c3 30. Bc4 c2 31. Rg7 c1=Q 32. Rf7 Qcc2 33. Bd3
Qb2 34. a3 b6 35. Kh4 Qd1 36. Rg7 Qdc1 37. Be4+ Kc5 38. Kxh5 Qd1+ 39. Bf3 Qdd2
40. Kg4 Qdc2 41. Kg5 Qd2+ 42. Kg4 Qf2 43. Kf4 Qxh2+ 44. Ke4 Qbc2+ 45. Ke3 Qhf2+
46. Kf4 Qce2 47. Kg4 Qd2 48. Rf7 Qb2 49. Kg5 Qfd2+ 50. Rf4 Qf2 51. Rxd4 Kxd4
52. Kh5 Kd3 53. a4 Qb1 54. Kg5 Qbf1 55. Kh5 Q1g1 56. Kh6 Qgf1 57. Kxh7 Q1g1 58.
Kh8 Qfh2+ 59. Bh5 Qhf2 60. a5 Qge1 61. Kg8 Qb1 62. a6 Qbe1 63. Kh8 Qb1 64. Kg8
Qbe1 65. Kh8 Qc1 66. Kg8 Qff1 67. Kh8 Kc3 68. Kg8 Qce1 69. Kh8 Qa1 70. Kg8 Qfc1
71. Kh8 Qab1 72. Bg6 Qbb2 73. Kg8 Qd1 74. Kf7 Qa2 75. Bh5 Qe1 76. Bg6 Qd1 77.
Bh5 Qe1 78. Bg6 Qee2 79. Ke8 Qab2 80. Kd8 Qf2 81. Kd7 Qfd2+ 82. Kc7 Qa2 83. Kc8
Qe2 84. Bh5 Qf2 85. Kb8 Qfd2 86. Bg6 Qdb2 87. Kc8 Qd2 88. Kc7 Qdb2 89. Kb7 Qd2
90. Kc6 Qdb2 91. Kc7 Qe2 92. Bh5 Qf2 93. Kb7 Qfd2 94. Bg6 Qdb2 95. Kc6 Qd2 96.
Kc7 Qe2 97. Kc8 Qad2 98. Bh5 Qc2 99. Bxe2 Kd2+ 100. Kb8 Qb2 101. Bd3 Kd1 102.
Kc7 Kd2 103. Kc6 Kc3 104. Be4 Qa2 105. Kc7 Qb2 106. Kb7 Kd4 107. Kxa7 Kxe4 108.
Kb7 Ke3 109. a7 Kd3 110. a8=Q Kc3 111. Qd8 b5 112. Qxe7 b4 113. Qc7+ Kd3 114.
Qd7+ Kc3 115. Qc7+ Kd3 116. Qd7+ Kc2 117. Qc7+ Qc3 118. Qd7 b3 119. Qe7 b2 120.
Qxe6 b1=Q+ 121. Qb6 Qc1 122. Qc6 Qb2+ 123. Qb6 Qa2 124. Qc6 Qb2+ 125. Qb6 Qd3
126. Qxb2+ Kxb2 127. Kc7 Qc3+ 128. Kb7 Ka2 129. Kb8 Qd3 130. Kc8 Qc3+ 131. Kb8
Qc2 132. Kb7 Qb2+ 133. Kc7 Qd2 134. Kb7 Qc2 135. Kb8 Qd2 136. Kc8 Qe2 137. Kb8
Qf2 138. Kc8 Qd2 139. Kb8 Qe2 140. Kc8 Qf2 141. Kb8 Qc2 142. Ka8 Qb2 143. Ka7
Qc2 144. Ka8 Qb2 145. Ka7 Qb1 146. Ka6 Qc1 147. Ka5 Qb2 148. Ka4 Qd2 149. Kb5
Qe2+ 150. Ka5 Qf2 151. Kb5 Qb2+ 152. Ka4 Qf2 153. Kb4 Qe2 154. Kc3 Qe1+ 155.
Kd4 Qf1 156. Ke3 Qe1+ 157. Kf3 Qh1+ 158. Ke3 Qg1+ 159. Kf3 Qb1 160. Kg2 Qc1
161. Kh2 Qd1 162. Kg2 Qe1 163. Kh2 Qf1 164. Kg3 Qe1+ 165. Kh3 Qf2 166. Kg4 Qg2+
167. Kh4 Qh2+ 168. Kg4 Qf2 169. Kh3 Qd2 170. Kg3 Qb2 171. Kh3 Qb1 172. Kh2 Qb2+
173. Kh1 Qb1+ 174. Kh2 Qe1 175. Kg2 Qd1 176. Kh2 Qe1 1/2-1/2"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(ROOT / "artifacts/runs/20260504_014403_e717/engines/step_00016000.trt"))
    ap.add_argument("--simulations", type=int, default=600)
    ap.add_argument("--cpuct", type=float, default=0.65)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=1, help="evaluate every Nth ply")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "artifacts/pgns/round6_analysis.csv")
    args = ap.parse_args()

    game = chess.pgn.read_game(io.StringIO(PGN))
    board = game.board()
    moves = list(game.mainline_moves())
    print(f"game has {len(moves)} half-moves")

    # Collect (ply_idx, fen, history_so_far, played_move) tuples
    samples = []
    history = []  # UCI moves
    for i, mv in enumerate(moves):
        # the position BEFORE this move was played is what we want to evaluate
        if (i % args.every) == 0:
            samples.append({
                "ply": i + 1,
                "fen": board.fen(),
                "history": list(history),
                "played_uci": mv.uci(),
                "played_san": board.san(mv),
                "stm": "W" if board.turn == chess.WHITE else "B",
                "move_num": (i // 2) + 1,
            })
        history.append(mv.uci())
        board.push(mv)

    print(f"evaluating {len(samples)} positions with MCTS-{args.simulations}...")

    # Batch through C++ MCTS
    histories = [s["history"] for s in samples]
    fens = [""] * len(samples)  # use empty FEN + full history (replay from startpos)
    t0 = time.time()
    results = _inference_cpp.small_mcts_search_batch_with_history(
        histories=histories,
        fens=fens,
        simulations=args.simulations,
        cpuct=args.cpuct,
        temperature=args.temperature,
        engine_path=args.engine,
        temperature_at_root=0.0,   # argmax over visits — what the model would PICK
        use_dirichlet=False,
    )
    dt = time.time() - t0
    print(f"done in {dt:.1f}s ({dt/len(samples)*1000:.0f} ms/position)")

    # Print per-ply table
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w") as f:
        f.write("ply,move_num,stm,played_san,played_uci,mcts_pick,W,D,L,W_stm,L_stm\n")
        print()
        print(f"{'ply':>4} {'mvN':>4} {'stm':>3} {'played':>10} {'mcts_pick':>10}   {'W':>5} {'D':>5} {'L':>5}   {'W(stm)':>7} {'L(stm)':>7}")
        print("-" * 90)
        for i, s in enumerate(samples):
            r = results[i]
            wdl = list(r["value"])  # [W, D, L] from side-to-move POV
            picked = r["action"] or "(none)"
            agree = "*" if picked == s["played_uci"] else " "
            # WDL from White's POV for clarity
            if s["stm"] == "W":
                w_stm, l_stm = wdl[0], wdl[2]
            else:
                w_stm, l_stm = wdl[0], wdl[2]   # WDL is already side-to-move POV
            print(f"{s['ply']:>4} {s['move_num']:>4} {s['stm']:>3} "
                  f"{s['played_san']:>10} {picked:>10}{agree}  "
                  f"{wdl[0]:>5.2f} {wdl[1]:>5.2f} {wdl[2]:>5.2f}   "
                  f"{w_stm:>7.2f} {l_stm:>7.2f}")
            f.write(f"{s['ply']},{s['move_num']},{s['stm']},{s['played_san']},{s['played_uci']},{picked},"
                    f"{wdl[0]},{wdl[1]},{wdl[2]},{w_stm},{l_stm}\n")
    print()
    print(f"saved CSV to {args.out_csv}")


if __name__ == "__main__":
    main()
