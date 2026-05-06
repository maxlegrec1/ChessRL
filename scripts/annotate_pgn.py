"""Write an annotated PGN with the model's WDL after each move + an eval plot.

Reads scripts/analyze_pgn.py's CSV and produces:
  - artifacts/pgns/round6_annotated.pgn  (PGN with {W=.. D=.. L=.. pick=..} per move)
  - artifacts/pgns/round6_eval.png       (W/D/L curves over plies, white POV)

The WDL after each move is "what the model thinks the position is worth from
the side TO MOVE on the next ply." We re-anchor everything to White's POV so
the curves are readable: W_white(ply) = W if stm=='W' else L.
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import chess
import chess.pgn

ROOT = Path("/mnt/2tb_2/ChessRL")
CSV_PATH = ROOT / "artifacts/pgns/round6_analysis.csv"
PGN_OUT = ROOT / "artifacts/pgns/round6_annotated.pgn"
PLOT_OUT = ROOT / "artifacts/pgns/round6_eval.png"

ORIGINAL_PGN = """[Event "MCTS brawl"]
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
    rows = list(csv.DictReader(CSV_PATH.open()))
    print(f"loaded {len(rows)} ply rows from {CSV_PATH}")

    # Index by ply (1-based)
    by_ply = {int(r["ply"]): r for r in rows}

    # Parse the PGN with python-chess and re-emit with comments
    game = chess.pgn.read_game(io.StringIO(ORIGINAL_PGN))
    new_game = chess.pgn.Game()
    new_game.headers.update(game.headers)
    new_game.headers["Annotator"] = "step_16000 MCTS-600"

    node = new_game
    for i, mv in enumerate(game.mainline_moves(), start=1):
        node = node.add_main_variation(mv)
        r = by_ply.get(i)
        if r:
            w = float(r["W"]); d = float(r["D"]); l = float(r["L"])
            pick = r["mcts_pick"]
            played = r["played_uci"]
            agree = "✓" if pick == played else f"≠{pick}"
            # WDL is in side-to-move-was-about-to-play POV, so it's "the side
            # that just moved sees the position from opponent's POV". Convert
            # to white's POV for clarity.
            if r["stm"] == "W":
                w_white, l_white = w, l
            else:
                w_white, l_white = l, w
            comment = f"W={w_white:.2f} D={d:.2f} L={l_white:.2f} | {agree}"
            node.comment = comment

    PGN_OUT.write_text(str(new_game) + "\n")
    print(f"wrote annotated PGN to {PGN_OUT}")

    # Eval plot — white POV across plies
    plies = sorted(by_ply.keys())
    w_curve, d_curve, l_curve = [], [], []
    advantage = []  # W_white - L_white
    for p in plies:
        r = by_ply[p]
        w = float(r["W"]); d = float(r["D"]); l = float(r["L"])
        if r["stm"] == "W":
            ww, ll = w, l
        else:
            ww, ll = l, w
        w_curve.append(ww)
        d_curve.append(d)
        l_curve.append(ll)
        advantage.append(ww - ll)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax = axes[0]
    ax.plot(plies, w_curve, color="steelblue", lw=1.5, label="W (white wins)")
    ax.plot(plies, d_curve, color="gray",      lw=1.5, label="D (draw)")
    ax.plot(plies, l_curve, color="crimson",   lw=1.5, label="L (black wins)")
    ax.set_ylabel("probability")
    ax.set_title("MCTS-600 root WDL after each move (white POV) — Round 6: step_10000 (W) vs step_15400 (B)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.plot(plies, advantage, color="black", lw=1.0)
    ax.fill_between(plies, advantage, 0, where=[a > 0 for a in advantage],
                    color="steelblue", alpha=0.3, label="white edge")
    ax.fill_between(plies, advantage, 0, where=[a < 0 for a in advantage],
                    color="crimson", alpha=0.3, label="black edge")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("ply")
    ax.set_ylabel("W − L (white POV)")
    ax.set_title("Eval advantage")
    ax.set_ylim(-1.0, 1.0)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=110)
    print(f"wrote plot to {PLOT_OUT}")


if __name__ == "__main__":
    main()
