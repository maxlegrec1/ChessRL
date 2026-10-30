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

## GPU session: where self-play time goes (2026-10-30)

A rented RTX 4090 with an EPYC 7742 (32 cores), TensorRT 10.14, branch `perf/cpu-bench`, run C's
network at step 92.5k (encoder 128 x 4, FP16 engine built as `scripts/build_trt.sh` does).
Self-play numbers are `scripts/bench_selfplay.py` with the recipe (90 s after 15 s of warm-up);
engine numbers are `trtexec` with CUDA graphs and no host copies. Scripts: `scripts/perf/`
(`perf_session.sh`, then `perf_quant.sh`); the logs are kept on the home machine in
`~/chessrl-runs/perf-2026-10-30/`.

The branch builds there, and its C++ tests (26) and GPU parity tests (20) pass.

**Self-play is GPU-bound.** With the recipe it makes 232,502 evaluations/s (422.8 recorded
positions/s) in batches averaging 113 positions, 0.95 ms each. The search threads spend 86% of
their time waiting for the network; the rest (selection, packing, expansion and backup, about
45 µs of thread time per evaluation on this CPU) keeps about 10 of the 32 cores busy.

**The engine alone** (positions/s by batch size): 124k at 32, 203k at 64, 275k at 128, 331k at
256, 312k at 512. Self-play gets 84% of the engine's speed at its batch size. At its best the
engine does about 33 TFLOP/s of useful work (about 0.1 GFLOP per position).

**Batching is already at its best:**

| setting | evaluations/s | mean batch | ms per batch |
|---|---|---|---|
| recipe: 128 threads, 4 inference workers | 232,502 | 113 | 0.95 |
| 2 inference workers | 154,301 | 123 | 0.69 |
| 64 threads | 212,378 | 60 | 0.60 |
| 256 threads | 228,139 | 193 | 1.86 |
| 512 threads (graph bucket 512 added) | 175,662 | 356 | 5.01 |

**Where the engine's time goes** (`trtexec --dumpProfile`, batch 128): 143 kernels per forward
pass, median 4.9 µs. Matrix multiplies (25 kernels, attention included) take 31% of the time;
fused elementwise kernels (norms, softmax, SwiGLU, reshapes; 118 kernels) take 69%. The
max-abs RMSNorm's reduction kernels alone are about 8%. Per-layer profiling inflates small
kernels, so the elementwise share is an upper bound, but the engine is bound by many small
kernels, not by arithmetic: faster matrix multiplies (INT8, FP8) can only speed up the 31%,
while fusing the elementwise work into the matrix multiplies (a fused kernel) addresses the 69%.

**Engine builds:** ONNX export 3.5 s, `trtexec` 82 s; with a timing cache reused, 11 s (14 s
while self-play runs). During a full build self-play slows by about 7% (97 s of build in a
150 s window: 222k evaluations/s over the window); one build per 500 training steps makes that
under 1% overall.

**Sharing the GPU with the trainer:** a synthetic trainer (the checkpoint's network, batch 1024,
AdamW) busy 15% of the time, about the real trainer's pace, cuts self-play to 207k evaluations/s
(-11%); busy 100% of the time, to 101k (-57%). NVIDIA MPS does not speed up self-play (229k
alone, 204k next to the 15% trainer) but runs the trainer's steps 31% faster next to self-play
(61 ms instead of 89 ms), which in a paced run would leave the GPU free sooner; a paced run
would show by how much.

**Precision and network size** (`scripts/perf/perf_quant.sh`): INT8 and FP8 engines from NVIDIA
ModelOpt 0.47 post-training quantization, calibrated on the 414 benchmark positions; the bigger
encoders have random weights, so only their speed means anything. Positions/s at each engine's
best batch size, and the useful arithmetic rate in FP16 (from an estimate of each network's
FLOPs):

| network | parameters | FP16 | INT8 | FP8 | FP16 TFLOP/s |
|---|---|---|---|---|---|
| 128 x 4 (today's) | 0.81M | 332k | 306k | 346k | ~33 |
| 192 x 6 | 2.51M | 146k | 118k | 162k | ~46 |
| 256 x 8 | 5.72M | 80k | 77k | 87k | ~58 |
| 384 x 10 | 15.8M | 42k | 44k | 47k | ~83 |

FP8 gains 4 to 14%, more on bigger networks; INT8 is slower at three of the four sizes (its
quantize and dequantize steps add kernels). Bigger networks use the GPU much better: 3x the
parameters (192 x 6) cost 2.3x the speed, 19x (384 x 10) cost 8x.

Accuracy of today's network quantized, on the 600 positions of the gap study, with the C++ search
at the gap study's settings (value errors in expected score; the FP16 row reproduces the gap
study's numbers):

| engine | value error vs Stockfish | 400-sim search loss | top move as FP16 | search value error |
|---|---|---|---|---|
| FP16 | 0.115 | 0.0246 | 100% | 0.084 |
| INT8 | 0.124 | 0.0256 | 84.5% | 0.090 |
| FP8 | 0.124 | 0.0258 | 79% | 0.097 |

Quantized this way, the value head is about 8% worse and the search's value up to 15% worse, for
at most 4% more speed on today's network.

**What this changes:**

- Batching: nothing to gain; keep 128 threads and 4 inference workers.
- Engine builds: a timing cache saves 70 s of GPU per build, under 1% of self-play; refit
  (updating the running engine's weights) matters more for memory than for speed.
- The trainer costs self-play about 11% of the GPU, and MPS does not get it back.
- Quantization: not worth it now. FP8 might be worth another look for much bigger networks, with
  quantization-aware training to keep the value head accurate.
- The engine is limited by its many small kernels (69% of its time), not by arithmetic: fusing
  them (a fused kernel, or norms TensorRT can fuse) is the remaining lever on the GPU side, with
  tree reuse (about 23% fewer evaluations) on the search side.
- Bigger networks cost much less than their size: the engine's efficiency rises from about 33
  to 83 TFLOP/s between 0.8M and 16M parameters.

## Tree reuse (2026-10-30)

`selfplay.tree_reuse` (off in the recipe until a GPU run measures it): a search hands back the
subtree of the move it plays (`MctsOptions::keep_tree`), and the game's next search continues it
(`reuse_tree`): `simulations` becomes the number of visits the root's moves end up with, so only
the missing ones are run. A full search mixes fresh Dirichlet noise into a reused root. With the
option off, searches are unchanged (same benchmark digests).

Checks with the CPU benchmark (fake network, sharpness 3):

- `bench_search --mode reusecheck` (412 positions, no root noise, most visited move): after a
  400-simulation search and its move, a search continuing the kept subtree (400 visits in all)
  picks the same move as a fresh 400-simulation search in 89.6% of positions, where a fresh
  800-simulation search does in 72.8%; value gap to the fresh 400 search 0.0063 against 0.0100.
  The root's moves end with exactly max(400, inherited) visits in every position. On average
  such a search inherits 161 of its 400 visits.
- `bench_search --mode selfplay` (the real `SelfPlayWorker`, recipe settings, 256 games, 150 to
  300 steps, three runs each): 137 to 139 network evaluations per move without reuse, 100 to
  102 with it (-27%), close to the -23% estimated with run C's network. CPU time per move rises
  by about 8% (subtree extraction, bigger trees to summarise). Games get longer (145 to 153 plies
  against 130 to 133): a fast move after a full search now has about 200 visits instead of 50,
  and a control without reuse but with 200-simulation fast moves also gives 146. The share of
  games drawn by repetition (10 to 16% with reuse, 15 to 16% without) varies as much between
  runs as between settings.

Not measured yet: the effect on training. Records from reused searches include visits gathered
before the root noise was mixed in, and games are longer. A GPU A/B at equal hours decides
whether `tree_reuse` goes into the recipe.
