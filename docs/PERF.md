# Search performance log

Two questions: what the self-play search costs on the CPU, and how many network evaluations a
good training target needs.

## CPU cost of the search

Measured with `scripts/bench_cpu_search.sh`: the production search code
(`cpp/mcts/mcts_small.cpp`) with a fake network in place of TensorRT
(`cpp/bench/fake_network.cpp`), on the 414 positions of `data/bench/positions.txt` (Stockfish 16
self-play from 40 balanced openings, plies 8 to 140, each with its real game history). Settings
are the recipe's: full searches of 400 simulations with Dirichlet noise, fast ones of 50,
cpuct 0.65, prior temperature 1.3, virtual-loss width 8, fixed seeds. The fake network's
sharpness 3 gives about 7 effective moves at the root (a trained network's priors at
temperature 1.3 are of that order).

"search µs/eval" is process CPU time per network evaluation, minus the fake network's own cost.
The digest hashes every search's root visit counts: a change meant to leave the search
unchanged must keep it.

Machine: AMD Ryzen 9 9950X3D, g++ 15.2, `-O3 -march=native` (as `cpp/setup.py`), 8 cores
(`taskset -c 0-7`), `nice 10`.

### Baseline (2026-10-29)

| search | threads | evaluations/s | search µs/eval | digest |
|---|---|---|---|---|
| full (400) | 1 | 147,699 | 6.65 | `4948b7eb443d881d` |
| full (400) | 8 | 1,033,428 | 7.13 | `4948b7eb443d881d` |
| fast (50) | 1 | 144,579 | 6.80 | `08ab46f09ebce77b` |
| fast (50) | 8 | 1,076,881 | 6.92 | `08ab46f09ebce77b` |

Where the time goes (`PHASES=1`, full, 1 thread, one pass of 164,566 evaluations): expansion and
backup 830 ms (84%), descent 105 ms (11%), packing and enqueue 39 ms (4%), the fake network's
answers 15 ms. Expansion copies the board, plays the move and runs `isGameOver()` (a full
legal-move generation) for every child of the expanded node.

With a simulated GPU (`EXTRA="--delay-us 500 --max-batch 128"`: batches of up to 128 answered
after 500 µs) and self-play's 128 blocking threads on the same 8 cores: 170,100 evaluations/s
(full), 15.5 µs of CPU per evaluation; fast 171,775/s, 12.3 µs. The threading model about
doubles the CPU cost per evaluation.

At self-play's ~250k evaluations/s, even 15 µs per evaluation is about 4 of a box's 32 cores:
the search's CPU cost is not what limits self-play today; the GPU is.

## Evaluations per training target: Gumbel vs PUCT (2026-10-29)

`scripts/search_quality.py` runs the CPU search (`src/analysis/engine.py`) and a Gumbel root
search (`src/analysis/gumbel.py`: Danihelka et al., ICLR 2022, with the reference
implementation's defaults, 16 candidates) on the 600 positions of the policy -> search gap study
(`~/chessrl-runs/runC-open0/gap_positions.json`, every legal move scored by Stockfish), with run
C's network. A target's sampled loss is the expected score lost, by Stockfish, when playing a
move drawn from it; the top-move loss is that of its favourite move. PUCT is self-play's
search (prior temperature 1.3, cpuct 0.65, virtual-loss batches of 8, moves-left tiebreak);
its target is the visit counts. Gumbel's target is its improved policy over all moves.

Check: the CPU PUCT at 400 simulations reproduces the C++ search's numbers from the gap study
on the same checkpoint (step 49.5k): sampled loss 0.0296 vs 0.0295, top-move loss 0.0134 vs
0.0139, 4.82 vs 4.79 effective moves; the raw policy matches exactly (0.0460, 0.0243, 6.40).

Step 92.5k (run C's newest network); 49.5k shows the same pattern:

| setting | evaluations | sampled loss | sampled moves losing > 0.10 | top-move loss | effective moves |
|---|---|---|---|---|---|
| raw policy | 1 | 0.0425 | 11.2% | 0.0176 | 6.3 |
| PUCT 50 (self-play's fast moves) | 51 | 0.0719 | 17.4% | 0.0147 | 10.1 |
| PUCT 100 | 100 | 0.0477 | 12.5% | 0.0113 | 7.3 |
| PUCT 200 | 198 | 0.0334 | 9.3% | 0.0103 | 5.8 |
| PUCT 400 | 394 | 0.0247 | 7.3% | 0.0117 | 4.7 |
| PUCT 400 + Dirichlet noise (the training targets) | 394 | 0.0258 | 7.4% | 0.0102 | 4.9 |
| Gumbel 8 | 9 | 0.0265 | 7.7% | 0.0129 | 3.4 |
| Gumbel 16 | 17 | 0.0254 | 7.5% | 0.0134 | 3.5 |
| Gumbel 32 | 33 | 0.0229 | 6.8% | 0.0114 | 3.3 |
| Gumbel 64 | 64 | 0.0210 | 6.4% | 0.0113 | 3.1 |
| Gumbel 128 | 127 | 0.0182 | 5.6% | 0.0100 | 2.7 |
| Gumbel 32, PUCT below the root | 33 | 0.0227 | 6.9% | 0.0117 | 3.4 |

Gumbel 16 to 32 matches PUCT 400's sampled loss with 12 to 24 times fewer evaluations, but its
targets are sharper (3.3 effective moves against 4.9), and a sharper target scores a lower
sampled loss whatever its quality. At equal sharpness (`scripts/search_quality_report.py`
sharpens each PUCT target set, p^(1/T), to the Gumbel setting's effective number of moves),
Gumbel is worth about as much as PUCT at a similar budget:

| step 92.5k | evaluations | sampled loss | PUCT 50 sharpened | PUCT 100 sharpened | PUCT 400 sharpened |
|---|---|---|---|---|---|
| Gumbel 32 | 33 | 0.0229 | 0.0223 (T 0.32) | 0.0191 (T 0.47) | 0.0164 (T 0.72) |
| Gumbel 128 | 127 | 0.0182 | 0.0186 (T 0.23) | 0.0157 (T 0.35) | 0.0138 (T 0.56) |

By the favourite move, Gumbel at 32 to 64 simulations is about as good as PUCT at 100. Gumbel
saves at most about 2x on these measures, not the 12x the first table suggests.

Two things the run also shows:

- **Target temperature.** Today's training target, PUCT 400 + noise, sharpened to T 0.7 has a
  sampled loss of 0.0155 instead of 0.0258 (-40%) at no evaluation cost. Whether a sharper
  target trains a better network (or an overconfident one) needs a training A/B.
- **Tree reuse.** Under PUCT 400 the move played holds half the root's visits (mean 0.50,
  median 0.48), 50 or more in 96% of positions; under PUCT 50, 26%. Keeping that subtree for
  the next search, with the budget counting the visits it already has (as KataGo does): a fast
  move after a full search is free 96% of the time, and with self-play's 25% full / 75% fast
  mix the expected evaluations per move fall from 137.5 to about 106 (-23%). The reused root
  needs fresh Dirichlet noise on full searches.
