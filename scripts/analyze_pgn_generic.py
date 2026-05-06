"""Same as analyze_pgn.py but reads PGN from a file. Outputs CSV +
annotated PGN + eval plot.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import chess
import chess.pgn

ROOT = Path("/mnt/2tb_2/ChessRL")
sys.path.insert(0, str(ROOT))

import _inference_cpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, type=Path)
    ap.add_argument("--engine", default=str(ROOT / "artifacts/runs/20260504_014403_e717/engines/step_00016700.trt"))
    ap.add_argument("--simulations", type=int, default=600)
    ap.add_argument("--cpuct", type=float, default=0.65)
    ap.add_argument("--out-prefix", required=True)  # written to artifacts/pgns/
    args = ap.parse_args()

    out_dir = ROOT / "artifacts/pgns"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / f"{args.out_prefix}_analysis.csv"
    annot_out = out_dir / f"{args.out_prefix}_annotated.pgn"
    plot_out = out_dir / f"{args.out_prefix}_eval.png"

    pgn_text = args.pgn.read_text()
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        print("failed to parse PGN", file=sys.stderr)
        return 1
    board = game.board()
    moves = list(game.mainline_moves())
    print(f"game has {len(moves)} half-moves")

    samples = []
    history = []
    for i, mv in enumerate(moves):
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

    print(f"running MCTS-{args.simulations} on {len(samples)} positions...")
    histories = [s["history"] for s in samples]
    fens = [""] * len(samples)
    t0 = time.time()
    results = _inference_cpp.small_mcts_search_batch_with_history(
        histories=histories, fens=fens,
        simulations=args.simulations, cpuct=args.cpuct, temperature=1.0,
        engine_path=args.engine,
        temperature_at_root=0.0, use_dirichlet=False,
    )
    print(f"done in {time.time() - t0:.1f}s")

    with csv_out.open("w") as f:
        f.write("ply,move_num,stm,played_san,played_uci,mcts_pick,W,D,L\n")
        print()
        print(f"{'ply':>4} {'mvN':>4} {'stm':>3} {'played':>10} {'mcts_pick':>10}   {'W':>5} {'D':>5} {'L':>5}")
        print("-" * 70)
        for i, s in enumerate(samples):
            r = results[i]
            wdl = list(r["value"])
            picked = r["action"] or "(none)"
            agree = "*" if picked == s["played_uci"] else " "
            print(f"{s['ply']:>4} {s['move_num']:>4} {s['stm']:>3} "
                  f"{s['played_san']:>10} {picked:>10}{agree}  "
                  f"{wdl[0]:>5.2f} {wdl[1]:>5.2f} {wdl[2]:>5.2f}")
            f.write(f"{s['ply']},{s['move_num']},{s['stm']},{s['played_san']},{s['played_uci']},{picked},"
                    f"{wdl[0]},{wdl[1]},{wdl[2]}\n")

    # annotated pgn
    new_game = chess.pgn.Game()
    new_game.headers.update(game.headers)
    new_game.headers["Annotator"] = f"step_{Path(args.engine).stem.split('_')[-1]} MCTS-{args.simulations}"
    node = new_game
    for i, mv in enumerate(moves):
        node = node.add_main_variation(mv)
        s = samples[i]
        r = results[i]
        wdl = list(r["value"])
        if s["stm"] == "W":
            ww, ll = wdl[0], wdl[2]
        else:
            ww, ll = wdl[2], wdl[0]
        agree = "✓" if r["action"] == s["played_uci"] else f"≠{r['action']}"
        node.comment = f"W={ww:.2f} D={wdl[1]:.2f} L={ll:.2f} | {agree}"
    annot_out.write_text(str(new_game) + "\n")
    print(f"\nwrote {csv_out}")
    print(f"wrote {annot_out}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plies = [s["ply"] for s in samples]
    w_curve, d_curve, l_curve, adv = [], [], [], []
    for i, s in enumerate(samples):
        wdl = list(results[i]["value"])
        if s["stm"] == "W":
            ww, ll = wdl[0], wdl[2]
        else:
            ww, ll = wdl[2], wdl[0]
        w_curve.append(ww); d_curve.append(wdl[1]); l_curve.append(ll); adv.append(ww - ll)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax = axes[0]
    ax.plot(plies, w_curve, color="steelblue", lw=1.5, label="W (white wins)")
    ax.plot(plies, d_curve, color="gray", lw=1.5, label="D")
    ax.plot(plies, l_curve, color="crimson", lw=1.5, label="L (black wins)")
    ax.set_title(f"MCTS-{args.simulations} root WDL — {game.headers.get('White')} (W) vs {game.headers.get('Black')} (B)")
    ax.set_ylabel("probability"); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(plies, adv, color="black", lw=1.0)
    ax.fill_between(plies, adv, 0, where=[a > 0 for a in adv], color="steelblue", alpha=0.3)
    ax.fill_between(plies, adv, 0, where=[a < 0 for a in adv], color="crimson", alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("ply"); ax.set_ylabel("W − L (white POV)"); ax.set_ylim(-1.0, 1.0)
    ax.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(plot_out, dpi=110)
    print(f"wrote {plot_out}")


if __name__ == "__main__":
    main()
