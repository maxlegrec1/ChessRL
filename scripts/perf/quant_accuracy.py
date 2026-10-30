"""Does a quantized engine play as well? (run on the box, project venv)
python quant_accuracy.py fp16=PATH int8=PATH fp8=PATH

On the 600 Stockfish-scored positions of the policy -> search gap study, with the C++ search as
in that study (400 simulations, prior temperature 1.3, cpuct 0.65, virtual-loss width 8, no
noise): the network's root value (0-simulation search) against FP16 and against Stockfish, and
the 400-simulation search's sampled and top-move losses and its top move's agreement with FP16."""

import json
import os
import sys

os.environ.setdefault("CHESSRL_MCTS_BATCH_SIZE", "64")
os.environ.setdefault("CHESSRL_CUDA_GRAPHS", "0")
os.environ.setdefault("CHESSRL_INFER_WORKERS", "2")

import _inference_cpp as cpp  # noqa: E402

pos = json.load(open("/workspace/perf/gap_positions.json"))
hist = [p["moves"] for p in pos]
sf = [p["scores"] for p in pos]
best = [max(s.values()) for s in sf]
engines = dict(a.split("=", 1) for a in sys.argv[1:])


def search(path: str, sims: int) -> list[dict]:
    return cpp.small_mcts_search_batch_with_history(
        histories=hist, fens=[""] * len(hist), simulations=sims, cpuct=0.65, temperature=1.3,
        engine_path=path, temperature_at_root=0.0, use_dirichlet=False, virtual_loss_width=8,
    )  # fmt: skip


def expected(wdl) -> float:
    return wdl[0] + 0.5 * wdl[1]


res = {}
for label, path in engines.items():
    v0 = [expected(r["value"]) for r in search(path, 0)]
    s400 = search(path, 400)
    dists = [{m: p for m, p in r["policy"] if p > 0} for r in s400]
    res[label] = (v0, dists, [expected(r["value"]) for r in s400])

ref_v0, ref_d, _ = res["fp16"]
ref_top = [max(d, key=d.get) for d in ref_d]
print(f"{'engine':<6} {'value vs fp16':>13} {'value vs SF':>11} {'search loss':>11} {'top loss':>8} "
      f"{'top = fp16':>10} {'search value vs SF':>18}")  # fmt: skip
for label, (v0, dists, vs) in res.items():
    n = len(pos)
    loss, top_loss, same = [], [], 0
    for i, d in enumerate(dists):
        s = sf[i]
        tot = sum(p for m, p in d.items() if m in s) or 1.0
        loss.append(sum(p / tot * (best[i] - s[m]) for m, p in d.items() if m in s))
        top = max(d, key=d.get)
        top_loss.append(best[i] - s.get(top, 0.0))
        same += top == ref_top[i]
    print(
        f"{label:<6} {sum(abs(a - b) for a, b in zip(v0, ref_v0, strict=True)) / n:>13.4f} "
        f"{sum(abs(a - b) for a, b in zip(v0, best, strict=True)) / n:>11.4f} "
        f"{sum(loss) / n:>11.4f} {sum(top_loss) / n:>8.4f} {same / n:>10.1%} "
        f"{sum(abs(a - b) for a, b in zip(vs, best, strict=True)) / n:>18.4f}"
    )
