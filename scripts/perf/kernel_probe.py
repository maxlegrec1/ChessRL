"""Where the fused kernel's time goes, on a GPU box (int8 branch experiments).

    uv run python scripts/perf/kernel_probe.py NET.cnet [NET.cnet ...] [--phases] [--modes] [--mocks]

--phases  the instrumented build (CHESSRL_KERNEL_TIMING=1): share of cycles per phase, and its own cost.
--modes   the accumulation modes at two blocks per SM (CHESSRL_FP16_ACC 256 / 64 / 0): what 32-bit sums cost
          (64: 32-bit accumulators, full-rate multiplies) and what the multiply rate is worth (0: half rate).
--mocks   the projection loop's timing mocks (CHESSRL_KERNEL_MOCK; wrong outputs, right timing): half the
          weight loads (1), half the multiplies (2), half the A loads (4), all three (7: what INT8 or FP8
          would do to the loop).
Each setting runs in its own process (the environment switches are read once), batch 128, 4 streams, best of 3.
"""

import argparse
import json
import os
import subprocess
import sys

BENCH = "import json, _inference_cpp as m; m.fused_phase_cycles(True); r = m.fused_benchmark({path!r}, 128, 400, 4); " \
        "print(json.dumps({{'rate': r, 'cycles': m.fused_phase_cycles(True)}}))"


def run(path: str, **env) -> dict:
    best = None
    for _ in range(3):
        out = subprocess.run([sys.executable, "-c", BENCH.format(path=path)], env={**os.environ, **env},
                             capture_output=True, text=True, check=True).stdout.splitlines()[-1]
        r = json.loads(out)
        if best is None or r["rate"] > best["rate"]:
            best = r
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nets", nargs="+")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--modes", action="store_true")
    ap.add_argument("--mocks", action="store_true")
    a = ap.parse_args()
    for path in a.nets:
        base = run(path)
        print(f"== {path}: {base['rate'] / 1e6:.3f}M positions/s")
        if a.phases:
            timed = run(path, CHESSRL_KERNEL_TIMING="1")
            total = sum(timed["cycles"].values())
            print(f"   phases (instrumented build {timed['rate'] / base['rate'] - 1:+.1%}):")
            for name, c in sorted(timed["cycles"].items(), key=lambda kv: -kv[1]):
                print(f"     {c / total:6.1%}  {name}")
        if a.modes:
            for acc, what in (("64", "32-bit accumulators, full-rate multiplies"), ("0", "FP32 sums, half rate")):
                r = run(path, CHESSRL_FP16_ACC=acc)
                print(f"   {what}: {r['rate'] / base['rate'] - 1:+.1%}")
        if a.mocks:
            for mock, what in (("1", "half the weight loads"), ("2", "half the multiplies"), ("4", "half the A loads"),
                               ("7", "all three (INT8/FP8 loop)")):
                r = run(path, CHESSRL_KERNEL_MOCK=mock)
                print(f"   mock {mock}, {what}: {r['rate'] / base['rate'] - 1:+.1%}")


if __name__ == "__main__":
    main()
