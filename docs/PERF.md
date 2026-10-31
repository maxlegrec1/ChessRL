# Search performance log

Two questions: what the self-play search costs on the CPU, and how many network evaluations a
good training target needs.

## CPU cost of the search

Measured with `scripts/bench_cpu_search.sh`: the production search code
(`cpp/mcts/mcts_small.cpp`) with a fake network in place of the GPU one
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
(`perf_session.sh`, then `perf_quant.sh`), removed with TensorRT (commit 0607bbf has them); the
logs are kept on the home machine in `~/chessrl-runs/perf-2026-10-30/`.

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

`selfplay.tree_reuse` (in the recipe since 2026-10-30): a search hands back the
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

**On a GPU** (`scripts/perf/tree_reuse_ab.sh`, now `selfplay_ab.sh`: RTX 4090 with an EPYC 7702P, run C's network at
step 92.5k, the recipe against the recipe with `tree_reuse: true`, alternated twice, 60 s of
warm-up and 240 s measured each):

| setting | recorded positions/s | games/s | evaluations/s | evaluations per position | mean batch |
|---|---|---|---|---|---|
| recipe | 452.8, 444.1 | 51.1, 49.7 | 248.7k, 244.6k | 550 | 117 |
| tree reuse | **685.0, 678.1** | 70.3, 69.7 | 270.1k, 267.7k | 394 | 115 |

Recorded positions per second rise by 52%. Two effects multiply: 28% fewer evaluations per
recorded position (394 instead of 550, alone worth +39%), and 9% more evaluations per second,
because most searches now start from a root that is already expanded and skip its separate
single-position network call. The GPU does the same work per batch (1.05 ms) but gets 10% more
batches per second, so it was waiting on the search part of the time. Games per second rise
less (+39%) because games are longer.

Not measured: the effect on training. Records from reused searches include visits gathered
before the root noise was mixed in, and games are longer. The recipe turned it on without a
training A/B (2026-10-30), for the self-play speed.

## The fused network and two CPU fixes (2026-10-30)

**The fused network** (`cpp/net/`, commit 0607bbf; TensorRT removed in 24b25f2): one kernel
launch per batch, one thread block per position (`docs/ARCHITECTURE.md`). Against PyTorch FP32
(run C at step 92.5k, 256 puzzle positions, `scripts/net_parity.py`): top-1 agreement 100%,
policy KL below 1e-5, WDL error 0.0006 on average, and the packed input path gives the dense
path's outputs exactly. Exporting a checkpoint takes 0.1 s (ONNX and `trtexec`: 85 s).

The kernel alone on an RTX 4090 (`_inference_cpp.fused_benchmark`, positions/s, output copy
included) against the TensorRT engine (`trtexec`, CUDA graphs):

| batch | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| TensorRT, 1 stream | 124k | 203k | 275k | 331k | 312k |
| fused, 1 stream | 223k | 422k | 813k | 839k | 860k |
| fused, 4 streams | 947k | 1,065k | 1,065k | 1,067k | 1,068k |

TensorRT at batch 128 on four streams: 294k. At 1.07M positions/s the kernel does about
106 TFLOP/s, 64% of the card's FP16 peak with FP32 accumulation.

**Lazy finished-game check** (65fd74b): expanding a node used to copy the board for each legal
move, play it, run `isGameOver()` (a full legal-move generation) and build a history string,
for children that are mostly never visited. The search now checks a node when a descent first
reaches it. Same digests on the 414 benchmark positions and on 605 positions 1 to 6 plies
before the end of games finished by mate, stalemate, repetition or the fifty-move rule (same
evaluation counts too), same `reusecheck` results. Search CPU per evaluation
(`bench_cpu_search.sh`, one thread): 6.82 to 1.94 µs (full searches), 7.08 to 2.00 µs (fast).

**Blocking sync** (a019d74, `CHESSRL_BLOCKING_SYNC=1`, off by default): the inference workers
sleep on a blocking-sync event instead of spinning in `cudaStreamSynchronize` while the GPU
runs their batch.

**Self-play on one box.** An RTX 4090 with a Ryzen 9 7950X (16 cores, a 30.7-core container
quota), rented whole. `bench_selfplay.py` now reports the CPU the rest of the host used during
each measurement: at most 0.1 cores in every arm. `scripts/perf/selfplay_ab.sh`, run C's network,
60 s of warm-up and 240 s measured, the eight arms alternated twice. The TensorRT arms ran from
commit 0607bbf (the last with both backends), the plain fused arms from 24b25f2 (before the CPU
fixes); one TensorRT run failed to start (a TensorRT internal error) and was run again at the
end. Ratios are to the mean of the two TensorRT recipe runs (534.5).

| arm | recorded positions/s | vs TensorRT recipe | evaluations/s | ms per batch | cores | CPU ms per position |
|---|---|---|---|---|---|---|
| TensorRT, recipe | 534.8, 534.2 | ×1.00 | 294k | 1.24 | 8.4 | 15.8 |
| TensorRT, tree reuse | 762.2, 760.0 | ×1.42 | 300k | 1.18 | 10.8 | 14.3 |
| fused, recipe | 1,196.1, 1,177.7 | ×2.22 | 653k | 0.19 | 19.5 | 16.5 |
| fused, tree reuse | 1,347.8, 1,343.5 | ×2.52 | 530k | 0.21 | 23.1 | 17.2 |
| fused + lazy check | 1,496.1, 1,524.3 | ×2.83 | 831k | 0.20 | 10.8 | 7.2 |
| fused + lazy check, tree reuse (the recipe since c4048b3) | **1,989.5, 1,989.4** | **×3.72** | 784k | 0.20 | 14.7 | 7.4 |
| + blocking sync | 1,457.3, 1,463.5 | ×2.73 | 804k | 0.22 | 9.1 | 6.2 |
| + blocking sync, tree reuse | 1,923.7, 1,920.0 | ×3.60 | 759k | 0.23 | 12.4 | 6.5 |

- With TensorRT the GPU is the limit: about 300k evaluations/s with or without tree reuse, so
  reuse gains only what it saves in evaluations per position (×1.4).
- The fused network makes a batch 6 times faster (0.20 ms against 1.2) and moves the limit to
  the search's CPU: with tree reuse, 23 of the 30.7 cores before the lazy check.
- The lazy check cuts the CPU per recorded position 2.3 times (16.5 to 7.2 ms) and gives tree
  reuse its room back (×1.32 on top of it, against ×1.13 without it).
- Blocking sync saves 1.7 to 2.3 cores but costs 3 to 4%: waking a sleeping worker adds about
  20 to 30 µs to every batch, and the CPU is no longer short. It stays off.
- At the recipe, neither resource is full: 14.7 of 30.7 cores, 784k of the kernel's 1.07M
  evaluations/s. The search threads wait most of the time: a self-play step (one move on each of
  the 1,024 boards) takes 129 ms for 94 ms of GPU work at full speed, the rest being the start
  and the end of each step, when few searches are left and batches shrink.

The same arms earlier on an EPYC 7702P box (Zen 2, a 30.7-core quota on a shared 128-thread
host), before the lazy check: TensorRT recipe 451.8 and 465.8, fused 771.8 and 787.4, TensorRT
with tree reuse 689.2, fused with tree reuse 1,181.8, the last one CPU-bound (26 of 30.7 cores).
A neighbour on that host loaded it to about 95 threads during one attempt and cut the results
by 40 to 70%; another rented "whole" machine (a Ryzen 9 7950X3D) had about 24 threads busy outside
our container with nothing of ours running. Logs: `~/chessrl-runs/perf-2026-10-30/fused-ab/`
and `perf4/`.

**Batching settings** (same box, the recipe with tree reuse, one run each, 60 s + 240 s):

| setting | recorded positions/s | mean batch | ms per batch | cores |
|---|---|---|---|---|
| recipe: 128 threads, 4 inference workers, 1,024 boards, 100 µs flush | 2,004.3 | 119 | 0.20 | 14.7 |
| 192 threads | 1,830.3 | 164 | 0.31 | 15.2 |
| 256 threads | 1,820.1 | 210 | 0.35 | 17.1 |
| 2,048 boards | 1,982.6 | 123 | 0.21 | 15.6 |
| 50 µs flush | 2,009.7 | 117 | 0.20 | 14.9 |
| 6 inference workers | 2,096.8 | 106 | 0.23 | 21.1 |

Bigger batches take proportionally longer and evaluations per second stay flat: the GPU is
nearly full (81 to 90% busy in `nvidia-smi dmon`). A batch of about 119 positions occupies 119 of
the 128 SMs (one block per position, one block per SM), so the workers' batches mostly queue
behind one another. Six workers overlap them a little more (+4.6%) for 6 more cores spinning on
the GPU; not adopted. The next self-play gains are in the kernel (FP16 accumulation in the
qkv, up and policy multiplies, smaller blocks so that two share an SM).

**With the trainer** (short real runs from run C's checkpoint, 12 minutes each, the last 8
measured; `docs/TRAINER_SPEED.md` has the trainer's side):

| setup | records/s | trainer steps/s | samples per record | GPU busy |
|---|---|---|---|---|
| self-play alone | 2,006 | | | 79% |
| trainer + self-play | 1,327 | 3.24 | 5.0 | 89% |
| both under MPS | 1,662 | 4.06 | 5.0 | 98% |
| MPS, trainer at below-normal priority | 1,964 | 3.46 | 3.65 | 99% |

Without MPS the two processes take turns on the GPU and self-play loses a third. Under MPS
their kernels run side by side; with the trainer at below-normal priority
(`CUDA_MPS_CLIENT_PRIORITY=1`) self-play loses 2% and the trainer runs in what is left, which
is not quite enough for the recipe's 5 samples per record.

## FP16 accumulation, MPS in the launcher, and where the inference workers' time goes (2026-10-30, evening)

Same box (RTX 4090 + Ryzen 9 7950X), run C's network, today's recipe (fused network, lazy
finished-game check, tree reuse).

**FP16 accumulation** (`CHESSRL_FP16_ACC`, the default since this change): the projection
multiplies (embedding, qkv, out, up, down, policy q and k) sum in FP16, twice the tensor-core rate
on GeForce cards, and add into FP32 every N terms; the attention and policy-logit multiplies and
the residual stream stay FP32. Parity against PyTorch FP32 (256 puzzle positions; run C at 92.5k,
base-wd at 91.5k, the 8-layer run A at 50k):

| accumulation | top-1 agreement | policy KL p95 | WDL error, mean | mlh max error |
|---|---|---|---|---|
| FP32 | 100 / 100 / 100% | 1e-5 | 0.0006 to 0.0008 | 0.0035 |
| FP16, 32 terms | 100 / 100 / 99.6% | 1e-5 | 0.0007 to 0.0010 | 0.0032 |
| FP16, 64 terms | 100 / 100 / 99.2% | 1e-5 | 0.0008 to 0.0013 | 0.0038 |
| FP16, whole sums (default) | 100 / 100 / 99.6% | 3e-5 | 0.0011 to 0.0015 | 0.0053 |

The kernel alone (4 streams, batch 128): 1,124k positions/s with FP32 accumulators, 1,352k with
64-term sums, 1,451k with whole sums (+29%). Self-play alone gains only 2.2% (2,008 to 2,052
recorded positions/s, two runs each), but next to the trainer the GPU time it frees goes to
training (below).

**MPS in the launcher** (`mps: true`, the recipe): `src.run` starts an MPS daemon for the run and
gives the trainer below-normal priority. Real runs from run C's checkpoint, 12 minutes, the last 8
measured:

| real run | records/s | trainer steps/s | samples per record | trainer waiting | GPU busy |
|---|---|---|---|---|---|
| no MPS (before) | 1,327 | 3.24 | 5.0 | 52% | 89% |
| MPS launcher, FP32 accumulation | 1,957 | 3.48 | 3.67 | 0% | 99% |
| MPS launcher, FP16 accumulation | **1,994** | **4.86** | **5.0** | 16% | 93% |

With both, self-play runs at its speed alone and the trainer reaches the recipe's 5 samples per
record again, with time to spare.

**Where an inference worker's time goes** (the runner now prints it when it closes; recipe,
4 workers, per batch of about 118 positions):

| | FP32 | FP16 accumulation | FP16, 8 workers |
|---|---|---|---|
| recorded positions/s | 1,999 | 2,048 | 1,977 |
| GPU and copies | 0.201 ms | 0.168 ms | 0.172 ms (75 positions) |
| results handed back | 0.295 ms | 0.310 ms | 0.303 ms |
| waiting for a batch | 0.058 ms | 0.058 ms | 0.209 ms |

The workers are busy 90% of the time and more than half of it goes to handing results back:
about 2.6 µs per position, which caps self-play near 800k evaluations/s whatever the GPU does.
Timed in two halves, filling the 1,929 FP32 logits (7.7 KB) into the result takes 1.3 µs per
position and waking the search thread (`std::promise::set_value`) 1.5 µs; converting alone takes
0.11 µs, so the filling cost is moving the memory between cores. More workers do not help (their
hand-back per position grows to 4 µs). A first fix, the search thread allocating the result so
the worker only fills it, was slower (1,851 against 1,999) and was dropped. What should work:
each search thread reads its record directly from the worker's pinned output buffer (the worker
writes nothing per position), and is woken once per batch rather than once per position.

**Zero-copy hand-back** (the next commit): each search thread reads its record straight from the
worker's pinned output buffer and converts it on its own core, and the worker wakes each search
once, when the last of its positions is back (`EvalGroup`), instead of once per position. The
worker never waits for a buffer to be read (a search holds the records already back while it
waits for the rest of its round, so waiting deadlocked the first version); with no free buffer
among its 16 it hands that batch back by value. Same digests on the CPU benchmark, in the
threaded mode too (128 threads, batches answered from another thread). Same box, today's recipe,
4 minutes per run:

| | recorded positions/s | evaluations/s | GPU and copies | handed back | waiting | cores |
|---|---|---|---|---|---|---|
| previous hand-back | 2,010.3, 2,062.0 | 802k | 0.168 ms | 0.31 ms | 0.06 ms | 15.5 |
| **zero-copy** | **2,373.6, 2,410.2** | 942k | 0.197 ms | 0.06 ms | 0.11 ms | 21.6 |
| zero-copy, 6 workers | 2,217.0, 2,210.2 | 870k | 0.199 ms | 0.05 ms | 0.22 ms | 22.2 |

+17.5%, ×4.47 the round-2 recipe on this box. The workers now wait 30% of the time; the GPU is
busier (more batches overlap, 0.197 ms per batch) and the search threads use more CPU (they
convert their own records). In a real run through the launcher self-play makes 2,347 records/s
(1,994 before), and the trainer is the limit again: 4.65 steps/s, 4.1 samples per record instead
of 5 (`docs/TRAINER_SPEED.md`).

**`torch.compile` in the trainer** (`compile: true`, the recipe since this change): the train step
is memory-bound element-wise work, 102 ms eager and 34 ms compiled on the 4090 alone
(`scripts/bench_trainer.py`, `docs/TRAINER_SPEED.md`). Real run through the launcher: self-play
2,402 records/s (its speed alone), the trainer 5.85 steps/s at 0.076 s per step, back to 5
samples per record and waiting half the time.

## The CPU scheduler as the limit (2026-10-31, night)

Same box (RTX 4090 + Ryzen 9 7950X, 32 hardware threads, a 30.7-core quota, nothing else
running), run C's network, self-play alone, `scripts/perf/selfplay_ab.sh` (60 s of warm-up,
240 s measured, the arms alternated twice) unless stated.

**Where the search threads' time goes** at the recipe (`CHESSRL_MCTS_PHASES=1`, 128 threads):
waiting for the network 53.6%, idle at the end of a step 27.1%, queueing evaluations 10.5%,
expansion and backup 6.0%, descent 2.8%. Less than a tenth of their time is search work. A step
(one move on each of the 1,024 boards) takes 106 ms: preparing the requests 1.4 ms, the searches
about 101 ms, applying the moves and writing the records 2.3 ms, Python 1.7 ms.

**More threads or boards, again** (self-play alone, 60 s + 180 s, one run each): they lost before
the zero-copy hand-back because the inference workers were full, and they still lose.

| setting | recorded positions/s | idle | waiting for the network | GPU per batch |
|---|---|---|---|---|
| recipe: 128 threads, 1,024 boards (two runs) | 2,415.1, 2,402.9 | 27.1% | 53.6% | 0.197 ms |
| 2,048 boards | 2,393.1 | 20.6% | 59.1% | 0.200 ms |
| 192 threads | 2,264.5 | 30.5% | 56.4% | 0.276 ms |
| 2,048 boards, 192 threads | 2,283.7 | 20.9% | 64.8% | 0.304 ms |

With 2,048 boards the threads are idle less, but the time they gain goes into waiting: the idle
end of a step is not what limits the speed.

**The run queue** (`/proc/schedstat`, the whole machine, 10 s at the recipe): 23.4 cores running,
and on average 22 threads runnable but waiting for a core, 432k context switches a second (192
threads: 29 waiting, 376k). Each search thread sleeps and wakes once per round of 8 evaluations,
about 120k wake-ups a second, in bursts as batches come back; part of "waiting for the network" is
waiting for a core once the results are back.

**Longest searches first** (fda7e84, reverted in 63be71b): the threads took the searches by
decreasing remaining simulations, so that no 400-simulation search starts at the end of a step.
2,310.5 and 2,318.3 against 2,400.8 and 2,411.6 (-3.8%): the idle fell from 27% to 19%, but with
every long search running at once more threads competed for the cores (queueing evaluations 14.4%
of thread time instead of 10.5%, a GPU batch 0.219 ms instead of 0.197).

**Batched enqueue** (e9def2e): a search queues its round of up to 8 positions under one lock, and
a worker is woken only when the queue was empty or just reached a full batch (each position used
to take the lock and wake a worker, which mostly found the batch still short and slept again).

**Interleaved searches** (aaba249, `CHESSRL_SEARCHES_PER_THREAD`, default 1): a thread keeps
several searches going and advances them in the order their evaluations were queued, so it waits
for the network only when its oldest round is not back. `SmallMctsRunner::run()` became
`start()`, `advance()` and `finish()`; each search does the same work in the same order (same
digests on the CPU benchmark with 1, 4 and 16 searches per thread, with and without a simulated
GPU, same `reusecheck` results, on this box too).

| arm | threads x searches | recorded positions/s | vs pushed code | cores | CPU ms per position | hand-back per batch |
|---|---|---|---|---|---|---|
| pushed code (0c274c1) | 128 x 1 | 2,391.4, 2,399.2 | | 22.9 | 9.5 | 0.057 ms |
| **batched enqueue** | 128 x 1 | **2,486.6, 2,494.4** | **+4.0%** | 23.0 | 9.3 | 0.061 ms |
| batched enqueue, interleaved | 32 x 4 | 2,437.2, 2,454.4 | +2.1% | 22.4 | 9.2 | 0.023 ms |
| batched enqueue, interleaved | 16 x 8 | 2,329.5, 2,331.8 | -2.7% | 14.7 | 6.3 | 0.010 ms |

- Batched enqueue: queueing evaluations falls from 10.8% to 2.7% of thread time and a step from
  106.8 to 102.7 ms.
- Interleaving does the same work with less CPU (16 x 8: a third less) and almost no run queue
  (host load 15 instead of 43), but 16 threads are then busy most of the time and become the
  limit. 47% of their time was outside every phase timer: taking a result (the FP16 record's
  conversion), `finish()` (the summary and the kept subtree) and freeing the tree, which the
  phases now time (771c75c).

**More splits, and the search threads at a lower priority** (the batched-enqueue build; nice 5
on the search threads so that a woken inference worker gets a core first, c14f37f removed it):

| arm | recorded positions/s |
|---|---|
| 128 x 1 | 2,495.4, 2,489.4 |
| 128 x 1, nice 5 | 2,495.8, 2,490.7 |
| 64 x 2 | 2,443.6, 2,442.9 |
| 32 x 4, nice 5 | 2,440.9, 2,443.8 |

Priority changes nothing, and interleaving loses 2% at every split: a thread advances its
searches one at a time, so a result that is back waits while the thread works on another search.
Interleaving stays available (`CHESSRL_SEARCHES_PER_THREAD`), off.

**The end of a search** (the new timers, 40 s + 120 s, one run each):

| | 128 x 1 | 16 x 8 |
|---|---|---|
| recorded positions/s | 2,487.6 | 2,295.3 |
| descent | 2.8% | 14.6% |
| queueing evaluations | 2.8% | 3.8% |
| expansion and backup | 6.5% | 24.2% |
| taking a result (`get`) | 1.9% | 6.6% |
| `finish()`: summary and kept subtree | 3.5% | 15.2% |
| destroying the search | 1.2% | 8.5% |
| waiting for the network | 59.4% | 9.7% |
| idle | 21.8% | 17.3% |

Taking the results, finishing and destroying a search cost 70% as much as the search's own
descents, queueing and expansions: `finish()` alone more than the descents. Each tree node holds
three vectors (children, their moves, their priors), so keeping the played subtree copies them all
and destroying a search frees them all. The next CPU saving is there (nodes whose children live in
one shared array, the kept subtree moved instead of copied), and `get` converts a whole 1,929-entry
FP16 record into a new vector when only the legal moves' logits are used.

**Batched enqueue is the recipe's code** (pushed, 771c75c): ×1.04 on the pushed code on this box,
×4.66 the round-2 recipe (534.5 to 2,490.5). Logs: `~/chessrl-runs/perf-2026-10-30/perf4/night.tgz`.

## ×8.6: the tree's allocations, the step barrier and the kernel's occupancy (2026-10-31)

A second box of the same kind (perf5: RTX 4090 + Ryzen 9 7950X, 32 hardware threads, no CPU quota,
rented whole), run C's network, self-play alone unless stated, `scripts/perf/selfplay_ab.sh` (60 s
of warm-up, 240 s measured, the arms alternated twice). This host runs the same code more slowly
than perf4 (CPU 12.1 ms per recorded position against 9.3), so every ratio below is measured on
perf5, against the round-2 recipe measured there too.

**The round-2 recipe on this box** (TensorRT, commit 0607bbf, the same alternation): 501.2 and
509.5 recorded positions/s (505.4). The code pushed before today (49dfae3) makes 2,059.6 and
2,063.0 here, ×4.08 (×4.66 on perf4).

**1. Flat tree nodes** (c76f8a2). Each node held three vectors (children, their moves, their
priors), allocated when it was expanded, copied with the kept subtree and freed with the tree. A
node's children are added together, so they are consecutive nodes: a node now holds `first_child`
and `num_children`, and each child its own move and prior. Self-play also stops building each root
move's principal variation, which it never read. Same digests on the CPU benchmark and on the real
network (414 positions, 400 simulations, no noise).

| arm | recorded positions/s | cores | CPU ms per position | step |
|---|---|---|---|---|
| pushed code | 2,055.5, 2,060.6 | 25.0 | 12.1 | 124 ms |
| **flat nodes** | **2,986.3, 2,982.7** | 14.8 | 5.0 | 86 ms |

+45%. The phases that allocate (expansion, taking a result, `finish()`, destroying the search) were
16 of the 23 points of search-thread time spent working, 71% of it; the CPU was the limit (about 29
threads wanting a core on 32 hardware threads, plus the spinning inference workers), so cutting the
CPU per position by 59% raised the speed until the GPU became the limit.

**2. Results read in place** (09d7e85): a result points at its FP16 record in the output buffer and
the search converts only the legal moves' logits (it allocated and converted all 1,929). +0.7%.

**3. The GPU's limit.** The kernel alone on this GPU (`fused_benchmark`): 925k positions/s at batch
128 on one stream, 1,384k to 1,409k on 4 or 8 streams at batch 64 to 256. Self-play evaluated
1.19M/s (84%), and `nvidia-smi dmon` showed the GPU running a kernel 87% of the time, with 4, 6 or
8 inference workers alike (3,017, 3,019, 3,006): the idle time came from the searches, not from the
workers. Within a step, the last long searches leave the GPU short of positions and the step's
serial part (preparing and applying 1,024 moves, 4.6 of 86 ms) leaves it with none.

**4. Asynchronous play** (bcfd343, `selfplay.async_play`, the recipe since 652d72d): persistent
search threads each play their own boards in turn and never wait for each other; a worker step
returns after one move per board's worth of moves. Preparing and applying a move run under one
lock; the searches run outside it.

| arm | recorded positions/s | evaluations/s |
|---|---|---|
| pushed flat nodes | 2,990.1, 2,985.9 | 1.18M |
| + results in place, step by step | 3,051.8, 3,051.8 | 1.20M |
| + longest searches first | 2,829.3, 2,827.7 | 1.11M |
| **+ asynchronous play** | **3,587.1, 3,595.8** | **1.415M** |

+17.7%, at the kernel's own ceiling: the GPU never waits. Longest-first loses again (it lost 4%
when the CPU was short; dropped in 6b1787a), and sleeping 100 µs before each sync (+0.9% step by
step) gains nothing once the GPU is the limit (dropped too).

**5. Two kernel blocks per SM** (f111865, the default since c64ee4a). The kernel used 92.5 KB of
shared memory and 224 registers per thread: one block (one position, 8 warps) per SM, and each of
its six barriers per layer idled the SM. Shared memory is now 41 KB (two FP16 buffers: the
attention output reuses the norm buffer once every warp has its q, k and v; the feed-forward runs in
two halves of 128 hidden columns through V's buffer; the policy's q and k reuse both), and the
default build is limited to 128 registers (216 bytes per thread spilled) so that two blocks share
each SM. Parity against PyTorch (256 puzzle positions, run C at 92.5k and the 8-layer run A at 50k)
is unchanged: top-1 100%, policy KL 1e-5, WDL error 0.0012 and 0.0014 on average. The down
projection's FP16 sums now cover 128 terms each half, so the search digest on the real network
changes (same accuracy).

| kernel alone (positions/s) | batch 128, 4 streams | 128, 8 | 256, 4 |
|---|---|---|---|
| pushed kernel | 1,304k, 1,409k | 1,392k, 1,405k | 1,410k, 1,406k |
| 41 KB, one block per SM | 1,408k, 1,409k | 1,405k, 1,404k | 1,406k, 1,406k |
| **41 KB, two blocks per SM** | **1,930k, 1,885k** | 1,852k, 1,841k | 1,854k, 1,875k |

| self-play (asynchronous play) | recorded positions/s | evaluations/s | cores |
|---|---|---|---|
| one block per SM, 4 workers | 3,581.7, 3,581.7 | 1.41M | 16.0 |
| two blocks per SM, 4 workers | 4,006.2, 3,989.5 | 1.58M | 20.8 |
| **two blocks per SM, 6 workers** | **4,362.9, 4,358.3** | **1.72M** | 26.3 |

With the faster kernel the 4 workers never wait for a batch (0.0006 ms per batch); 6 keep the GPU
busier (+9%), the default since c64ee4a.

**Where it ends** (this box): 4,360.6 recorded positions/s, **×8.63 the round-2 recipe** (505.4),
×2.12 the code pushed before today. Not kept: interleaved searches (32 x 4, 16 x 8: the same speed
with less CPU, only in the step-by-step mode; `CHESSRL_SEARCHES_PER_THREAD`, off).
