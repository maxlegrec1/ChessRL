"""Hard Lichess puzzles with a checkpoint on the CPU, strict line (every model move must match).

For every model move of each puzzle: the solution move's raw-policy probability, its share of
the search's visits, and the network's favourite raw move. One JSON row per puzzle (resumable).
Used to pick the blog's puzzle (docs/blog/make_boards.py).

  CUDA_VISIBLE_DEVICES= uv run python scripts/puzzle_scan.py --ckpt step_00100500.pt --out puzzles.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.engine import Net, analyse  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-rating", type=int, default=2500)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    net = Net(a.ckpt, threads=a.threads)
    rows = [r for r in csv.DictReader((ROOT / "data/puzzles/lichess_1k.csv").open())
            if int(r["Rating"]) >= a.min_rating]
    done = {json.loads(line)["id"] for line in a.out.read_text().splitlines()} if a.out.exists() else set()
    for r in sorted(rows, key=lambda r: -int(r["Rating"])):
        if r["PuzzleId"] in done:
            continue
        board = chess.Board(r["FEN"])
        moves = r["Moves"].split()
        board.push_uci(moves[0])  # the opponent's move that sets up the puzzle
        steps, ok = [], True
        for i in range(1, len(moves), 2):
            res = analyse(net, board, a.sims)
            want = moves[i]
            by = {m["uci"]: m for m in res["moves"]}
            raw_top = max(res["moves"], key=lambda m: m["prior"])
            steps.append({"fen": board.fen(), "want": want, "san": board.san(chess.Move.from_uci(want)),
                          "played": res["best"], "prior": round(by[want]["prior"], 3) if want in by else 0.0,
                          "share": round(by[want]["share"], 3) if want in by else 0.0,
                          "raw_top": raw_top["san"], "raw_top_prior": round(raw_top["prior"], 3),
                          "wdl": [round(x, 3) for x in res["wdl"]]})
            if res["best"] != want:
                ok = False
                break
            board.push_uci(want)
            if i + 1 < len(moves):
                board.push_uci(moves[i + 1])
        row = {"id": r["PuzzleId"], "rating": int(r["Rating"]), "themes": r["Themes"], "fen": r["FEN"],
               "moves": moves, "solved": ok, "sims": a.sims, "ckpt": a.ckpt, "steps": steps}
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(r["PuzzleId"], r["Rating"], "solved" if ok else "missed",
              [(s["san"], s["prior"], s["share"]) for s in steps], flush=True)


if __name__ == "__main__":
    main()
