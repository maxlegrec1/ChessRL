# Next round: faster, cleaner self-play learning

> **Historical log.** The experiment configs (`configs/*.yaml`) and the one-off study scripts
> named below were removed from main once the round ended; the winning recipe became
> `src/train/config.yaml`. They are all at commit
> [`ff17fbf`](https://github.com/maxlegrec1/ChessRL/tree/ff17fbf260f0c09fea29615fb62b82d93ab95194),
> the merge of this round.

Started 2026-10-21. Budget: **$200 of rented GPU time** (Vast.ai), tracked by the
`guard` command of `scripts/vast/vast.py`. This file is the plan and the lab notebook —
results get appended to the log at the bottom as they land.

## Results summary (final, 2026-10-24 08:30 UTC)

Spent **$148.70** of $200: 15 rented RTX 4090 boxes, 243 box-hours. Ladder Elo
is anchored on calibrated Stockfish/random/greedy players (random = 0), ±40
per rating; hours are hours of training (downtime excluded).

| recipe (one 4090 each) | ~900 Elo | 1,500 | 1,800 | at 8 h | plateau |
|---|---|---|---|---|---|
| `main` (original code) | ~20.5 h | — | — | 233 | 921 at 21.7 h, still rising |
| + learning-signal fixes (ResNet) | ~3 h | ~16 h | — | 1,234 | ~1,500 at 21 h |
| + 0.81M transformer encoder, 800/100 sims | 2.9 h | ~4 h | 8.0 h | 1,800 | ~1,950 from ~13 h |
| + 400/50 sims (best per hour) | **2.3 h** | **3.5 h** | **6.0 h** | **1,847** | ~1,860-1,900 from ~11 h |

What mattered and what didn't:
- **Policy targets = MCTS visit distribution** is the dominant fix (−730 Elo
  without it at 6M positions); searching with outcome-trained value heads is
  an early accelerator; opening-only temperature is neutral; infrastructure
  alone buys throughput, not learning.
- **Encoder > ResNet**: +450-580 Elo at equal positions despite −32%
  self-play throughput.
- **400/50 simulations** beat 800/100 per hour and replicate (seed 2: 1,503
  vs 1,504 at 3.5 h); **200/25 is too cheap** (−100 to −250 from 3 h,
  flattens ~1,400).
- **Growth**: ResNet growth a modest +0-90 Elo from 8 h (puzzles 60% vs 54%);
  encoder growth from 2 layers or 8 layers from scratch: level per position,
  behind per hour; encoder grown 4 -> 6 -> 8 at its plateau (final run,
  step 67k): **~1,970**, no gain beyond noise. LR decay and resignation
  calibration branches: same ~1,950.
- **The "collapse" at 17 h was an FP16 bug**, not learning: the hand-written
  RMSNorm overflowed in the TensorRT engine once activations passed 256
  (no weight decay). Fixed (54ac876); re-rated, E's checkpoints never declined
  (45.5k = 1,989). `scripts/trt_parity.py` checks engine vs PyTorch.
- **Weight decay** (AdamW 0.05, matrices only) keeps the residual stream
  bounded at equal strength: max activation into the final norm 46 vs 171 at
  30k steps (weight norm 99 vs 164); 1,818 vs ~1,820. **Continued overnight
  to 18.6 h (step 73k): it does not break the plateau.** Mean Elo by hours,
  decay vs no decay (sims-400 seed 1): 8-10 h 1,800 vs 1,835; 12-14 h 1,836
  vs 1,877; 14-16 h 1,829 vs 1,876; 16-18 h 1,864 vs 1,890; 18-20 h 1,888
  (puzzles equal, ~72%). Weight norm flat (104 at 50k, 107 at 70k) vs 206 ->
  224 without; max activation 80 at 70k vs 441 at 60k. A flat weight norm
  means a constant effective LR, so "the net slowly freezes" is ruled out as
  the plateau's cause. Final spend **$148.70**, 15 boxes, 243 box-hours.

Tooling added on the way: Vast launch/deadline/guard/sync CLI (`chessrl-vast`),
per-run anchor ladder, combined dashboard with run toggles, keep-alive resume
on boxes, and a CPU analysis board (`scripts/analysis_server.py`, dashboard
page "Analysis"). Blog draft: `docs/blog/chessrl-next-round.mdx`.

## Why (audit of `main` @ 128893c)

| Finding | Where | Consequence |
|---|---|---|
| Policy target is the one move *sampled* at τ=1; the visit distribution is discarded | `cpp/selfplay.cpp` (`record.policy = summary.action`) | one noisy sample per position instead of ~30 visit fractions |
| Root τ=1 for the whole game, no resign, 600-ply cap | `config.yaml`, `cpp/selfplay.cpp` | game outcome `z` reflects sampling noise; z/q agree ~67% |
| Policy loss weight 0.1 vs 1.0 + 1.0 for the value heads | `config.yaml` | policy starved of gradient |
| Random-replacement 2M buffer + 200k random-net warmup | `cpp/selfplay.cpp`, `config.yaml` | stale/garbage data lingers; slow start |
| Trunk is ~0.36M params (4×64); policy FC is ~0.76M | `src/models/small/small_model.py` | capacity is tiny; GPU compute is ~2% of inference latency, so growth is cheap |
| Absolute board orientation | `src/models/small/fen_encoder.py` | net learns both colour perspectives (speedrun measured ×1.53 for canonical) |
| Python per-sample batch build, full-buffer re-parse every 20 s, synchronous TRT build | `src/train/` | trainer stalls; becomes the bottleneck once self-play speeds up |
| 8 unused heavy deps (torchtune, torchao, torch-tensorrt, modelopt, onnx-graphsurgeon, onnxruntime, pycuda, onnxscript) | `pyproject.toml` | slow installs on rented boxes |

## Plan

Every new behaviour is behind a config flag whose "off" value reproduces the
old behaviour, so A/B arms run the same binary.

1. **Signal fixes** — record v2 (current bitboards + top-K visit distribution),
   soft policy loss with equal weights, τ cutoff after N plies, resignation with
   a no-resign fraction, playout-cap randomization, FIFO buffer with a growing
   sampling window, vectorized loader, async engine build.
2. **Growth** — function-preserving depth/width growth of the ResNet trunk
   (zero-init residual blocks), triggered on plateau or schedule.
3. **Encoder** — a *small* speedrun-style encoder (canonical board, x0
   re-injection, QK-norm, geometric attention bias, attention policy head),
   sized to the self-play budget (~1–3M params), paired with fewer sims.
4. **Hyperparameter search** — successive halving over short fixed-budget runs,
   scored against a fixed Stockfish ladder + puzzles (not the self-relative chain).
5. **Final long run** with the winning recipe.

## Budget ($200)

Rented boxes: RTX 4090 / 5090 with ≥32 effective CPU cores (MCTS is CPU-bound);
~$0.55/h for a 4090 on 2026-10-21. H100/H200 are a poor fit (~$3.4–4/h, ≤28 cores).

| Phase | Boxes × hours | Est. |
|---|---|---:|
| Bring-up, build, parity tests, size/throughput bench | 1 × 16 | $9 |
| A/B: baseline vs signal fixes | 2 × 24 | $27 |
| Growth + encoder arms | 3 × 24 | $40 |
| HP search (successive halving) | ~6×8 + 3×8 | $40 |
| Final run | 1 × 96 | $55 |
| Reserve | | $29 |
| **Total** | | **$200** |

The guard refuses new launches above $180 committed and destroys every
instance at $200.

## Measurement protocol (for the write-up)

- **Arms.** A = `main@128893c` untouched (`box.sh main-setup` / `main-run`);
  B = `configs/main.yaml` (main's recipe on the new infrastructure);
  C = `configs/signal.yaml` (B + learning-signal fixes); later D = growth,
  E = encoder, then the tuned recipe.
- **Hardware.** One single-GPU box per arm (RTX 4090/5090, >=32 cores). The
  speedup is reported as two factors measured separately: learning efficiency
  (Elo vs self-play positions, hardware independent) and throughput
  (positions per hour; `scripts/bench_sizes.py` on one box + per-arm logs).
- **Strength.** Anchor ladder: random mover, greedy capture, node-limited
  Stockfish at several skills plus skill-0 Stockfish mixed with random moves —
  node limits keep it CPU-speed independent. Anchors are calibrated once by a
  round robin (random = 0, `src/rating/anchors.json`). Every 30 min (same
  cadence in every arm) the newest engine plays 48 paired-opening games
  (MCTS-100) against the two anchors nearest its last estimate; Elo is the MLE
  against fixed anchor ratings, with a standard error. Plus strict-line
  accuracy on 1,000 Lichess puzzles (200 per rating band).
- **Headline.** Time (and positions) to first reach fixed ladder thresholds,
  per arm, with uncertainty from the ladder standard errors. Raw JSONL for
  every chart is synced home to `~/chessrl-runs/<box>/`.

## Phase 2 (approved 2026-10-22 ~08:30 UTC)

| What | Where / when (UTC 10-22) | Ends | Est. |
|---|---|---|---|
| Ablations: −value, −policy, −temp (`configs/ablate_*.yaml`, 6 h) | 3 new 4090 boxes, ~09:00 | +7 h deadline | $17 |
| Ablations: −resign, −pcr, −buffer | armA 20:50, armB 21:08, armD 22:29 (after their 24 h) | +6.5 h | $10 |
| C + cosine LR decay (`signal_decay.yaml`) | dev1 21:10 | +10.5 h | $6 |
| Final: E continued, 4 -> 6 layers + cosine (`encoder_final.yaml`) | armE 23:12 | +17 h | $10 |

Projected campaign total ≈ $110 of $200. Boxes on a shared host get disjoint
32-CPU blocks (`chessrl-vast cpus`); deadlines are enforced by the guard.

### Early ablation read (2026-10-22 10:55 UTC, ±~100 Elo noise; C vs D-before-growth as the seed replicate)

Elo at N self-play training records (positions axis — the ablation boxes have
faster CPUs than C's, so hours would flatter them):

| positions | 1M | 2M | 4M |
|---|---|---|---|
| C signal | 409 | 525 | 1023 |
| D (replicate of C until 8k steps) | 282 | 455 | 1026 |
| C − value fix | 126 | 277 | — |
| C − visit-distribution policy target | 206 | 285 | — |
| C − opening-only temperature | 436 | 699 | 1072 |

Value fix and policy target clearly carry weight early; the temperature cutoff
does not (whole-game sampling is as good or better early — more diverse data
may matter more than clean outcomes at this stage). Also: the encoder arm's
ladder daemon died at ~9 h (TensorRT OOM next to a 14.7 GB trainer); restarted
with a batch-256 runner.

### Ablations at 6 h (2026-10-22 14:51 UTC, positions axis)

| positions | 1M | 2M | 3M | 4M | 6M | 8M | 10M |
|---|---|---|---|---|---|---|---|
| C signal | 409 | 525 | 863 | 1023 | 1198 | 1222 | 1334 |
| D (C replicate until 8k steps) | 282 | 455 | 675 | 1026 | 1270 | 1398 | 1426 |
| E encoder | 415 | 462 | 901 | 1470 | 1651 | 1800 | 1809 |
| C − value fix | 126 | 277 | 504 | 932 | 1137 | 1292 | 1362 |
| C − opening-only temperature | 436 | 699 | 944 | 1143 | 1130 | 1274 | 1306 |
| C − visit-distribution policy target | 206 | 285 | 355 | 380 | 466 | 724 | — |

- **Policy target (visit distribution at equal weight) is the dominant fix:**
  ~600 Elo behind without it at 6-8M positions, and the gap is not closing.
- **Value fix: an early accelerator.** −250 at 2M, caught up by ~10M.
- **Opening-only temperature: neutral.** Whole-game sampling led early
  (more diverse data) but the edge was gone by 6M. The final run keeps the
  opening-only setting (no change).
- The −temp box's container exited on its own at ~5.8 h (data synced to
  14:30 UTC); destroyed.

### Collapse / scaling health check (2026-10-22 ~11:30 UTC, ~14 h in)

| signal | C signal | E encoder | read |
|---|---|---|---|
| policy entropy | 2.93 -> 2.05 | 2.82 -> 1.77 | smooth decline, no collapse |
| search-target entropy | 2.11 -> 1.29 (flat since 6 h) | 2.22 -> 1.30 | stable |
| draws | 32% -> 7% | 25% -> 7% | no draw death |
| castle rate | -> 1.1% | -> 1.1% | healthy (canonical 0.8-1.2%) |
| games ending by resignation | 66% | 74% | games ~55-59 plies: few endgames |
| wrong resignations (playthrough audit) | 7.4% | 7.3% | above the 5% AlphaZero target |

The −policy ablation (weak net, rated ~355) reached 15.4% wrong resignations
with the same fixed threshold: a weaker value head makes a fixed resign
threshold worse, another reason to calibrate it.

Response: resignation auto-calibration to <= 5% (`resign_target_fp`),
plateau-triggered growth, hold-then-decay LR, and watchdog alarms on
target entropy < 0.8, decisive < 30%, wrong resignations > 10%. All in the
final run (`configs/encoder_final.yaml`).

## Phase 3: encoder only (2026-10-22 18:45 UTC)

The ResNet arms had answered their questions (signal fixes, growth, which fix
matters) by ~20 h, and the encoder beat every ResNet arm from hour 3 on, yet it
was the only encoder run. The planned ResNet ablations (−resign, −pcr, −buffer)
and C's LR-decay continuation were cancelled; A, B, C, D stopped at 21 h.

**Why E stopped gaining (diagnosis at step 45k, 19.5 h):**

| steps | 5k | 10k | 15k | 20k | 30k | 40k | 45k |
|---|---|---|---|---|---|---|---|
| policy loss | 2.00 | 1.83 | 1.81 | 1.79 | 1.77 | 1.78 | 1.80 |
| KL(search target ‖ net) | 0.60 | 0.55 | 0.52 | 0.49 | 0.47 | 0.46 | 0.48 |
| value loss | 0.70 | 0.70 | 0.68 | 0.67 | 0.66 | 0.65 | 0.69 |
| ladder Elo | 576 | 1652 | 1576-1745 | 1809 | 1840 | 1904 | 1820 |

- Learning stopped at ~12k steps (5 h): the policy loss moved 0.04 in the
  next 30k steps and the net still sits 0.46 nats from its own search's
  visit distribution. Elo kept creeping (+100 from 8 h to 16 h) on that flat
  loss, then stalled.
- The LR was a constant 1e-3 (Adam, no weight decay) for all 45k steps, no
  decay ever, and the model is 0.81M parameters. Both are textbook causes
  of this curve; the run cannot tell them apart, so they are now tested as
  separate branches of the same checkpoint.
- Self-play is GPU-bound (96% utilisation; trainer steps take 0.2 s of every
  ~1.4 s, pacing at k = 5 samples per record), so a bigger net costs
  self-play speed one-for-one.
- Late drift from ~17 h: draws 7.0% -> 9.7%, games 51 -> 58 plies, puzzles
  73% -> 66% (1,000 puzzles: ~3.5 sigma), training losses rising. Nothing
  changed on the box (only self-play, trainer, ladder running). A
  constant-LR net drifting toward drawish play is the early shape of a draw
  collapse; watched in the branches.

**Phase 3 runs (all encoder):**

| box | run | question |
|---|---|---|
| armE | E @ 45k + cosine LR decay only (`encoder_decay.yaml`) | is the plateau the LR noise floor? |
| dev1 | E @ 45.5k copy + plateau growth + LR hold/decay + resign calibration (`encoder_final.yaml`), fresh buffer | is it capacity? (final run) |
| armA | from scratch, 400/50 sims (`enc_sims400.yaml`) | more games vs deeper search |
| armB | from scratch, 8 layers (`enc_deep.yaml`, 1.47M) | capacity from step 0, at −39% self-play |
| armD | from scratch, 2 -> 4 -> 6 -> 8 layers at 3k/9k/16k steps (`enc_grow.yaml`) | progressive growth vs starting big |
| armF (new) | from scratch, `encoder.yaml` again | seed noise: the yardstick for every comparison |
| armG (new, 19:15) | E @ 45.5k + resignation calibration only (`encoder_resigncal.yaml`), fresh buffer | is it value-target noise from wrong resignations? |

Third candidate cause, found at 19:08: right after the branch, E's net resigned
wrongly 10.2% (armE) and 12% (dev1) of the time at the fixed 0.05 threshold,
vs 7.3% at 14 h. A wrong resignation labels a drawn or won position as lost,
and rising draws make "looks lost, isn't" positions more common: a plausible
feedback loop behind the late decline. armE stays LR-only so each branch
changes one thing. armG (California, EPYC 9374F) started 19:22 UTC at 257
pos/s; first calibration window 8.4% wrong -> threshold 0.04.

Early-phase watch (19:15): enc-sims400's decisive-outcome share fell
0.69 -> 0.24 in its first 1.75k steps (47% draws, mostly repetition) while
value accuracy rose 0.49 -> 0.77: the early "shuffle and repeat" phase, deeper
than E's (0.55 at the same stage) as 400/50 sims find fewer mates. Collapse
alarms now apply from step 4k; compared step-for-step with seed 2 at 20:15.

20:15 check (decisive share / draw-labelled share of training records):

| steps | 1k | 2k | 3k | 4k | 5k | 6k |
|---|---|---|---|---|---|---|
| E (orig) | .52/.03 | .49/.11 | .65/.12 | .78/.07 | .83/.02 | .83/.01 |
| enc-sims400 | .43/.02 | .27/.19 | .45/.24 | .58/.11 | .67/.06 | .76/.04 |
| encoder-s2 | .53/.03 | .65/.11 | .64/.10 | | | |

sims-400 recovered: same shape as E, deeper dip, no collapse; ladder
131 -> 248 -> 349 -> 476 by step 5.5k (E at 5k: 576), slightly ahead per hour.

Ladder outage 19:00-20:10 UTC: the daemons of the 6-layer branch (dev1) and the
8-layer run (armB) died with TensorRT OOM (runner: one context per worker per
CUDA-graph bucket, 400 MB each at batch 256, next to ~20 GB of self-play +
trainer). All ladders now run at batch 64, 2 workers, no graphs, in a restart
loop (`box.sh ladder`). First ratings after branching (±42): E-decay 1,694
(step 48.1k, LR ~0.77x), growth 1,713 (47.5k), resign-cal 1,744 (46.8k): all
within ~2 SE of the branch point's 1,750-1,790.

**20:30 UTC: the plateau is value-target noise from wrong resignations, and
LR decay alone makes it worse.**

| since branch | resign threshold | wrong resigns | draws | plies/game | value loss | ladder |
|---|---|---|---|---|---|---|
| LR decay (armE) | fixed 0.05 | 11.3% | 7.9% | 57 | 0.68 -> 0.76, rising | 1,694 -> 1,590 |
| growth (dev1) | calibrated -> 0.028 | 5.6% | 15.9% | 77 | 0.75 -> 0.72 | 1,713 |
| resign cal (armG) | calibrated -> 0.020 | 5.5% | 15.1% | 75 | 0.74 -> 0.73 | 1,744 -> 1,762 |

At the fixed threshold about 1 resignation in 9 is wrong, and each labels a
drawable position as lost; a more pessimistic value head then resigns more
(7.3% wrong at 14 h, 11.3% now): the noise compounds. Lowering the LR fits that
noise more precisely, hence the faster decline. With calibration the threshold
has to fall to ~0.02-0.03, games get 20 plies longer and twice as drawn (many
"lost" games were not), and the value metrics recover. Next: armE restarts
from step 45.5k as calibration + LR decay (`encoder_caldecay.yaml`) once one
more rating confirms the decline, so armG / armE / dev1 = calibration alone /
+ decay / + growth.

**21:30 UTC correction: the real cause is FP16 overflow in the deployed
engine, not resignations.** All three branches kept declining (cal + growth
1,713 -> 1,578, cal only 1,762 -> 1,704, puzzles 61% -> 55-57%), so the cause
had to be shared. Measured on E's checkpoints (fp32, CPU, 512 puzzle
positions; `artifacts/scratch/act_scale.py`):

| step | weight norm | max \|x\| into final norm | max token RMS |
|---|---|---|---|
| 10k | 105 | 46 | 10.6 |
| 20k | 133 | 101 | 22.0 |
| 30k | 158 | 194 | 39.9 |
| 40k | 180 | 280 | 58.3 |
| 45.5k | 192 | 349 | 63.8 |

No weight decay, so the residual stream grows. The encoder's hand-written
RMSNorm (`x.pow(2).mean()`) runs in FP16 inside TensorRT, and x^2 overflows
past |x| = 256: the token's norm becomes inf, its output 0, silently.
`scripts/trt_parity.py` (FP16 engine vs PyTorch FP32):

| step | top-1 agreement | policy KL mean / p95 | max value error |
|---|---|---|---|
| 10k-30k | 99.2-100% | 0.0000 / 0.0001 | <= 0.012 |
| 40k | 98.8% | 0.071 / 0.0001 | 0.053 |
| 45.5k | 96.5% | 0.216 / 1.31 | 0.102 |
| 45.5k, fixed norm | 99.6% | 0.0000 / 0.0001 | 0.007 |

Self-play, the ladder and the puzzles all ran the drifting engine from ~40k
(E's decline started at 17 h = step ~40k); the trainer never saw it. The
rising wrong-resignation rate was most likely a symptom (the engine's values
drifted from the trained net's), not the cause. Fix (54ac876): divide by the
row max before squaring, the same function up to eps; deployed to every box
21:20 UTC, so each run's next engine export (every 500 steps) is clean. ResNet
arms are unaffected (no RMSNorm). Weight decay would also bound the growth;
candidate for the final recipe. Re-rated with fixed engines (armG, same ladder, 96 games, 21:20 UTC):

| step | corrupted engine: Elo / puzzles | fixed engine: Elo / puzzles |
|---|---|---|
| 30k (13 h) | 1,839-1,898 / 71% | 1,947 +-40 / 71.9% |
| 40k (17 h) | 1,904 / 71% | 1,892 +-40 / 72.7% |
| 45.5k (19.5 h) | 1,750-1,790 / 64-66% | 1,989 +-40 / 72.6% |

**The trained net never declined**; the whole "decline" was the engine. The
branch point (45.5k) is E's best checkpoint, so the branches continue from it.
E does still flatten from ~13 h (1,900-2,000, puzzles flat at ~72%): the
LR-vs-capacity question stands, at a higher level, and the branch comparison
restarts from the fixed engines (their buffers hold corrupted-engine games
until ~23:30 UTC; ratings before ~21:20 measured broken engines). Void: "LR
decay alone makes it worse" and "wrong resignations cause the plateau".

22:30 UTC report (fixed engines from ~21:15; branch buffers still flushing):

| branch @ 45.5k (1,989) | ratings on fixed engines (puzzles) |
|---|---|
| cal only (armG) | 1,834 -> 1,948 (71%) |
| cal + growth, 6 layers (dev1) | 1,821 -> 1,897 (73-74%) |
| cal + LR decay (armE) | 1,893 -> 1,913 (71%) |

| from scratch at ~3.5 h | Elo by hours | Elo by self-play records |
|---|---|---|
| E (orig) | 1,303 at 3.4 h | 1,303 at 3M, 1,652 at 4M |
| seed 2 (armF) | 1,296 at 3.5 h | 793 at 3M, 1,296 at 4M |
| sims 400/50 (armA) | 1,504 at 3.5 h | 1,082 at 4M, 1,504 at 6M |
| 8 layers (armB) | 459 at 2.8 h | 459 at 1M |
| grow 2->8 (armD) | 423 at 2.9 h | 454 at 2M |

Seed 2 tracks E closely in hours (first noise read). Sims-400 leads by ~200 at
equal hours (twice the games), about level per record. 8 layers and grow are
behind in hours but level per record, on the slower host.

21:52-22:45 UTC: armB (8 layers) self-play died swapping to its first FP16-safe
engine: TensorRT asked 723 MB for an execution context (sized for the batch-1024
max profile) while the old engine's contexts were still alive. Engines now build
with max batch 512 (self-play batches stay <= 256, the ladder runs at 64); armB
resumed on a rebuilt step-2.5k engine, deadline extended 1 h to 04:10 UTC.
Its step-3k swap at 23:25 went through (18.0 GB).

00:00 UTC report (buffers flushed of broken-engine games):

| branch @ 45.5k | ratings since 21:15 (puzzles) | LR now |
|---|---|---|
| cal only (armG) | 1,834 1,948 1,846 1,907 1,968 (73%) | 1e-3 |
| cal + LR decay (armE) | 1,893 1,913 1,927 1,941 1,896 (71 -> 76%) | 1.7e-4 |
| cal + growth (dev1) | 1,821 1,897 1,872 1,890 1,913 (73-74%), 6 layers | 6.4e-4 |

E's own fixed-engine level is ~1,940 (30k 1,947 / 40k 1,892 / 45.5k 1,989,
+-40 each): no branch is clearly above it after 3 h; LR decay leads on puzzles
(76%, the best of any run). dev1 grew 6 -> 8 layers at step 53.4k (23:44 UTC,
1.47M params; swap fine at 17.8 GB, self-play 250 -> 133 pos/s).

| from scratch | Elo at 5 h | at 2M / 3M / 4M / 6M / 8M records |
|---|---|---|
| E (orig) | 1,589-1,633 | 462 / 1,303 / 1,652 / 1,576 / 1,768 |
| seed 2 | 1,547 | 438 / 793 / 1,412 / - / - |
| sims 400/50 | 1,718 | - / - / 1,082 / 1,504 / 1,718 |
| 8 layers | 474 at 4.4 h | 459 at 1M (90 pos/s, lost 50 min) |
| grow 2->8 | 480 at 4.6 h | 454 / 480 |

Sims-400 is ~100 Elo ahead of E and seed 2 at equal hours (twice the games),
level per record. Seed 2 and E agree in hours within ~50-100. The two
slow-host runs are level with E per record so far but have only 1-3M records:
both still sit before the 2-4M-record jump E and seed 2 made.

01:26 UTC: armB's self-play died again, CUDA error 901 (stream capture
invalidated) launching the new norm's fused kernel right after an engine swap:
a CUDA-graph capture on one inference worker racing another worker's launch.
Only armB (8 layers, slowest) hit it in ~5 h of swaps on seven boxes. armB
resumed 01:51 with CUDA graphs off (`EXTRA_ENV=CHESSRL_CUDA_GRAPHS=0`), deadline
06:15 UTC (+2 h: it was mid-jump, 474 -> 920). Every run now has a keep-alive
(`box.sh keep`): a dead run resumes from its newest checkpoint, up to 5 times.

**03:00 UTC: from-scratch results at 8 h** (single runs; seed 2 = noise yardstick)

| run | 1,000+ | 1,500 | 1,800 | at 8 h |
|---|---|---|---|---|
| E, 800/100 sims | 3.4 h (1,303) | ~4.5 h | 8.0 h | 1,800 |
| seed 2, 800/100 | 3.5 h (1,296) | ~5 h | - | 1,699 at 7.5 h |
| **sims 400/50** | **2.5 h (1,082)** | **3.5 h** | **6.0 h** | **1,847** |
| grow 2 -> 8 (slow host) | 6.1 h | 7.1 h | - | 1,573 at 7.6 h |
| 8 layers (slow host, 2 crashes) | 8.0 h | - | - | 1,304 |

Sims-400 is the best from-scratch recipe: the same Elo per self-play record as
E (8M: 1,718 vs 1,768; 10M: 1,822 vs 1,829) at twice the records per hour.
Growth from 2 layers and 8 layers from step 0 are level with seed 2 per record
but lose in hours (slow host; the 8-layer net also costs 39% of self-play).

Branches at ~5.75 h on fixed engines (mean of the last ~5 ratings): cal + LR
decay ~1,960 (puzzles 75-76%, LR now at its 5e-5 floor), cal only ~1,950
(72-76%), cal + growth to 8 layers ~1,940 (72-74%; 8 layers since 23:44). E's
own fixed-engine level is ~1,940-1,990. **Over ~6 h none of LR decay, growth
or resignation calibration moves the plateau beyond the +-40 noise.**

Next (03:05 UTC): armA runs sims 200/25 from scratch (`enc_sims200.yaml`, same
box as sims-400, to 11:35 UTC); armF runs a second seed of sims 400/50 from
04:05 (to ~12:35). dev1's final run continues to 11:50, armB to 06:12, armG to
~05:10; armD (grow) and armE (cal + decay) end at their deadlines.

04:34 UTC: sims-200 resigns wrongly 15.4% of the time at 1.5 h (fixed 0.05
threshold, net at ~350): a quarter of the search gives noisier values, the same
rate the weak -policy ablation reached. Left as is (comparability with
sims-400); watchdog line raised to 20%. armB's 8-layer net: 1,400 at step 5.7k
(E needed ~7-8k steps for that): faster per step, slower per hour.

08:00 UTC: **sims-400 replicates** (seed 2 on armF: 1,503 at 3.5 h and 1,619
at 4.0 h vs seed 1's 1,504 / 1,614). **Sims 200/25 is too cheap**: 1,011 /
1,245 / 1,394 / 1,343 / 1,381 at 2.5-4.5 h vs sims-400's 1,082 / 1,382 / 1,504 /
1,614 / 1,638 on the same box; far behind per record (10M: 1,343 vs 1,822).
Stopped at 4.5 h; armA now runs sims-400 + AdamW weight decay 0.05 on
matrices (`enc_sims400_wd.yaml`, 08:02-16:33 UTC). Does sims-400 pass 2,000 if
continued? E gained ~150-190 after 8 h (re-rated: ~1,950 at 13 h, ~1,990 at
19.5 h) and sims-400 was ~50 ahead of E at 8 h with ~1.6x the steps per hour:
plausible, right at the ~1,950 ceiling every E branch sits at. Answered with
the best run itself (08:19 UTC): seed 1 resumed on armA from step 32.5k with
its own 2M-record buffer (stopped 03:05-08:19; dashboard hours now exclude
downtime), deadline 16:50 UTC. Seed 2 stopped at 4.2 h (replication done);
armF runs the weight-decay test instead (same box as seed 2, 08:20-16:51 UTC).

20:42 UTC: decline confirmed (LR-decay branch 1,583 at step 49k, LR 0.60x);
stopped. 20:46: armE restarted from step 45.5k as calibration + LR decay
(`caldecay`, fresh buffer like the other two); first window 8.0% wrong ->
threshold 0.04.

The growth branch starts on an empty buffer: E's 2.5 GB buffer would have
taken ~70 min through this machine (0.6 MB/s down from armE). E's 2M-record
FIFO turns over every ~2 h anyway, so the difference is confined to the
branch's first ~2 h. The ladder history was copied so the plateau trigger
fires at once (E: 1,904 -> 1,820 over its last 2 h).

From-scratch runs get 8.5 h each and are compared with E at equal hours and
positions (E: 901 at 2.9 h, 1,470 at 3.9 h, 1,652 at 4.4 h, 1,800 at 8 h).
Deep-encoder bench (armB, 32 CPUs, 800 sims): enc128x4 281 pos/s, x6 215,
x8 171; all fit in TensorRT memory next to the trainer.

Start-up (19:03 UTC, random-init nets, trainer + ladder running): armF
(Ryzen 7950X) 284 pos/s at 4 layers; armD 200 at 2 layers and armB 104 at 8
layers (both on the slower Thailand host, ladder Stockfish on the same 32
cores). Box speed differs by ~1.5x, so from-scratch runs are compared on the
**positions axis** first, hours second. Growth branch: 4 -> 6 layers (1.14M)
at its first step as expected; its first resignation window measured 12%
wrong at threshold 0.05, calibrated to 0.04.

## Log

- 2026-10-21 — audit done, branch `worktree-next-round` created.
- 2026-10-21 — dev box `dev1` (RTX 4090, 32 cores, $0.61/h). C++ built first
  try on TensorRT 10.14 (apt now defaults to TRT 11, pinned back); 69/69
  tests pass on GPU; 3,000 C++-written v2 records validated against
  python-chess (`scripts/check_buffer.py`).
- Bugs found in main while wiring this up:
  1. Every engine swap rebuilt the self-play worker, discarding all 1,024
     games in progress (records are only flushed at game end) — long games
     could never finish. Now `SelfPlayWorker.set_engine()` swaps in place.
  2. Trainer pacing slept ~410 s on its first step (rate estimated from an
     empty window). Now a token bucket: target_k samples per new record.
- Top-20 visit entries kept only 74% of the visit mass (p10 58%) under an
  untrained net — records widened to 84 entries (1,280 bytes).
- Smoke run (signal config, untrained net, 4090): ~450-500 recorded
  positions/s (25% full searches), ~500k nodes/s, 13-16 games/s; policy loss
  6.41 -> 4.29 in 200 steps; in-place engine swap at step 200 confirmed;
  first ladder rating 55 ± 42 Elo, puzzles 6.3% (68 s per rating).
- Anchor calibration: Stockfish skill 0 at 1/10/50 nodes all land within
  ~50 Elo (skill caps depth), so random-move mixtures fill the greedy ->
  Stockfish gap. Final ladder: random 0, sf_s0_r50 504, greedy 561,
  sf_s0_r25 778, sf_s0_n50 1224, sf_s5_n300 1428, sf_s10_n1000 1712,
  sf_s20_n1000 2068, sf_s20_n5000 2420, sf_s20_n20000 2663.
- Size bench (4090, 800 sims, 32 cores): 4x64 633 pos/s, 6x96 564,
  8x128 471, 10x128 456, 12x160 232, 16x192 181 — growth stops at 10x128.
- Phase 1 arms started: A (main as-is, armA, CPUs 32-63 — shares a host with
  dev1) 20:44; B (infra) 21:02; C (signal) restarted 21:06 after the fix below.
- **Bug 3 (main): the search value never sees game results.** main exports
  value_q / mlh_q to the engine, heads trained only on the search's own
  backed-up values. First 45 min of arm C: 98.7% of search-WDL targets had
  max prob < 0.4 (mean entropy 1.092 vs ln 3 = 1.0986), value_q loss pinned at
  1.0977, while game outcomes were informative (W 25% / D 51% / L 24%). A
  fixed point grounded only by mates inside the tree. `search_heads: outcome`
  feeds MCTS the result-trained value / true-plies mlh instead (AlphaZero).
  The 45-minute pre-fix run is kept as `signal-v0`.
- First ~1.2 h of phase 1 (ladder Elo, ±44 each, 48 games vs 2 anchors):
  | arm | t≈0 | t≈0.5 h | t≈1.0-1.5 h |
  |---|---|---|---|
  | A main as-is | 91 | 89 (step 500) | 43 (step 1000) |
  | B infra | 23 | 18 (step 1500) | 86 (step 4000) |
  | C signal | 97 | 190 (step 500) | 409 (step 2000) |
  C's search-value loss left ln 3 (0.797) while B's stayed at 1.096; C's
  puzzle first-move accuracy 17% -> 26%, A and B fell to 8-9%. Right after
  its first engine swap, A's self-play write rate dropped to ~4 records/s
  (all games restarted — bug 1 in action).
- 2026-10-22 00:20 (Paris): growth arm (armD, Thailand 64-core, pinned 0-31)
  started 22:23 UTC; D's first 8k steps double as a seed replicate of C.
  Two boxes (Spain, Texas) never finished pulling the CUDA image (stuck
  "loading" 18+ min; Vast bills only disk while loading) — destroyed.
- Encoder bench (armE: 4090, EPYC 7742, pinned to 32 CPUs): ResNet 4x64
  605 pos/s; enc128x4 (0.81M) 409 (-32%); enc192x4 (1.78M) 259 (-57%);
  enc192x6 OOMs creating TensorRT contexts (649 MB each at batch 1024).
  Attention over 64 tokens is costlier per parameter than the ResNet's convs
  at this size, so the encoder arm uses 128x4. Started 23:09 UTC.
- ~~Core-sharing incident~~ (corrected 08:45 UTC): armB and armD share a
  public IP but are **different machines** (Vast machine_id 30113 / 28904),
  so they never shared cores; D's live re-pin to 32-63 at 01:38 was a no-op.
  B's self-play did fall from 561 to 422 pos/s after ~1.3 h, but the likely
  cause is the net itself: sharper priors -> deeper trees -> more CPU per
  simulation (C slowed similarly). Only dev1/armA truly share a machine
  (10236), pinned 0-31 / 32-63 from the start. Pinning now groups boxes by
  machine_id, not IP.
- t ≈ 2.5-5 h (01:38 UTC): A 99, B 117, C 1100 (step 9.5k), D 867 (step
  7.5k, seed replicate of C until its 8k growth: C was 863 at 6.5k), E 462
  (step 4k). Puzzle line accuracy at step ~4k: C 25%, D 23%, E 47% — the
  encoder is far stronger tactically per step at equal ladder Elo. Low-end
  ladder caveat: deterministic MCTS-100 eval draws many won games by
  repetition vs weak anchors (C @3k: 20W 28D 0L vs random), compressing
  the low end of the scale equally for all arms.

## Value-head study, step 0 (2026-10-24 22:00 UTC, one rented 4090, $0.34)

Question: should the search read the head trained on the search's root WDL
(`value_q`) instead of the one trained on game results (`value`), at least late
in training? Test-time only: same checkpoint, three engines that differ only in
the value the search reads (moves-left fixed on the outcome head), from the
sims-400 run. `scripts/head_ab.py`; raw results in `~/chessrl-runs/headab/`.

| | step 10k | step 61.5k (plateau) |
|---|---|---|
| ladder, `value` (288 games) | 1,494 ± 36 | **1,885 ± 23** |
| ladder, `value_q` | 1,480 ± 35 | 1,808 ± 23 |
| ladder, 50/50 blend | 1,459 ± 34 | 1,850 ± 23 |
| head-to-head `value` vs `value_q` (400 games) | +16 (95%: −8 to +40) | **+54 (+30 to +79)** |
| head-to-head `value` vs blend | +4 (−20 to +28) | **+44 (+19 to +69)** |
| puzzles (line), value / value_q / blend | 58.7 / 59.7 / 59.8% | 72.6 / 72.7 / 72.8% |
| Spearman vs Stockfish WDL, value / value_q (2,000 positions) | 0.841 / 0.840 | 0.882 / 0.889 |

- At the plateau the search plays **50-80 Elo stronger reading the game-result
  head**, although `value_q` is as good or slightly better as a static estimate
  (rank correlation with Stockfish's WDL, every game phase). At step 10k the
  heads are level. So switching at test time does not help; the gap even grows
  with training.
- Likely reasons (untested): the policy, search settings and data were all
  trained with `value` in the search (a test-time switch pays the mismatch),
  and `value_q` learns a visit-weighted root average, i.e. the value of the
  position if play followed the search's visit distribution, which is softer
  than best play (root average sits 0.016 below the most-visited move's value).
  A training-time switch (step 1) is still untested; if run, use the
  most-visited move's value as the target.
- **Dirichlet noise barely moves the `value_q` target** (mean |shift| 0.004-0.005
  in expected score, mean shift −0.001; the 0.016 root-vs-best gap is the same
  with and without noise). It does change the most-visited move in 16-20% of
  full-search positions (`scripts/search_determinism.py`: 0% without noise, so
  it is the noise); the median value lost is 0.001 (near-ties), but 6-11% of
  those changes lose > 0.10, i.e. ~1-2% of full-search moves are noise-induced
  mistakes that end up in the games.
- **Ladder lag in steep climbs**: the run's own ladder rated step 10k at 1,082,
  against anchors it beat almost every game (greedy 561, sf_s0_r25 778); with
  better-matched anchors (778 and 1,224) the same engine rates ~1,490, and its
  puzzle accuracy is identical (59%). The daemon's anchors trail the estimate,
  so ratings during fast rises are lower bounds; this applies to every run, most
  to the fastest risers.

## Self-play move quality and the policy -> search gap (2026-10-25 18:00-19:40 UTC, on run C's box)

Both judged by Stockfish (MultiPV over every legal move, 300k nodes, WDL as an expected
score for the side to move); "loss" = expected score lost against Stockfish's best move.
`scripts/move_quality.py`, `scripts/policy_gap.py`; positions: 600 from 100-game samples
of runs A, B and C (300 from plies 0-29, 300 later); raw results in
`~/chessrl-runs/runC-open0/gap.jsonl`.

**Opening moves are mostly sampled from 50-sim searches, and those are worse than the raw
policy.** In plies 0-29 every move is sampled in proportion to the visit counts, and 75%
of moves get the quick 50-sim search, whose visits are far flatter than the policy
(~11 effective moves vs ~6: prior temperature 1.3, virtual-loss width 8, few sims). Run C,
step 49.5k, 300 opening positions:

| quick-move choice | mean loss | loss > 0.10 | loss > 0.30 | effective moves |
|---|---|---|---|---|
| 50-sim visits, sampled (today) | 0.098 | 25.0% | 13.3% | 11.2 |
| raw policy, T = 1.0 | 0.055 | 15.8% | 6.5% | 6.1 |
| raw policy, top-p 0.90 | 0.044 | 13.4% | 4.7% | 4.2 |
| raw policy, T = 0.7 | 0.038 | 11.9% | 4.0% | 3.7 |
| raw policy, top-3 | 0.036 | 10.9% | 3.5% | 2.3 |
| raw policy, T = 0.5 | 0.031 | 9.9% | 2.9% | 2.6 |
| top move (policy or 50-sim) | ~0.024 | 8.0% | 1.7-2.7% | 1 |

The moves actually played in C's games (0.086 mean loss, 12% > 0.30) match the 75/25 mix of
50-sim and 400-sim sampling: ~3 moves losing > 0.30 per game in the first 30 plies, so most
games are decided by an opening blunder (median 35 plies, 28/100 end within 30 plies).
This also explains run B (sampling all game from the same visit counts).

**Policy -> search gap along two runs** (600 positions, sampled-move loss):

| run | step | ladder Elo | policy | 400-sim search | gap | value head MAE | search value MAE |
|---|---|---|---|---|---|---|---|
| C | 5k | ~600 | 0.147 | 0.106 | 0.041 | 0.315 | 0.291 |
| C | 10k | ~1,420 | 0.086 | 0.056 | 0.030 | 0.223 | 0.187 |
| C | 20k | 1,669 | 0.057 | 0.034 | 0.023 | 0.160 | 0.127 |
| C | 30k | 1,754 | 0.052 | 0.034 | 0.019 | 0.140 | 0.117 |
| C | 40k | 1,792 | 0.047 | 0.030 | 0.018 | 0.132 | 0.102 |
| C | 49.5k | 1,825 | 0.046 | 0.030 | 0.017 | 0.123 | 0.104 |
| base | 10k | (1,082, ladder lag) | 0.076 | 0.044 | 0.032 | 0.198 | 0.160 |
| base | 37.5k | 1,857 | 0.050 | 0.028 | 0.021 | 0.128 | 0.102 |
| base | 45k | 1,839 | 0.050 | 0.029 | 0.021 | 0.122 | 0.102 |
| base | 52.5k | 1,890 | 0.047 | 0.031 | 0.016 | 0.125 | 0.109 |
| base | 61.5k | 1,890 | 0.045 | 0.030 | 0.015 | 0.120 | 0.097 |

- The search still removes a third of the policy's loss on the plateau (33-43% at every
  checkpoint), so the targets keep teaching; the policy slowly closes on them.
- **What flattens is the search itself**: its loss stops at ~0.03 from ~step 30-40k in both
  runs, and so does the value head's error (~0.12 MAE against Stockfish). The plateau looks
  like a value-quality ceiling: the search can only be as good as the values it backs up,
  and the value head learns from game results that opening blunders make noisy.
- As a run-health signal: watch the search's own loss and the value head's error; when both
  are flat, more training only lets the policy catch up the remaining gap (~0.015).

## Run C continued with policy-sampled quick moves (2026-10-25 19:30 -> 2026-10-26 05:30 UTC)

C was resumed from step 54k (its plateau) with `quick_policy_temperature: 0.7`
(`configs/enc_sims400_open0_qpol.yaml`): cheap moves in the first 30 plies sample the raw
policy at T = 0.7 instead of the visit counts of a 50-sim search. B was stopped to pay for it.

| | before (steps 40-54k) | after (steps 74-92k) |
|---|---|---|
| ladder Elo (mean of 8 / 10 ratings) | ~1,850 | ~1,940 |
| plies per game | 55 | 80-86 |
| resignations / draws | 76% / 6% | 66-68% / 14-16% |
| self-play positions per second | 273 | 424-456 |

On the cached Stockfish positions (all 600), the quantities that had stalled on the plateau
moved: the 400-sim search's move loss 0.0295 (C @ 49.5k) / 0.030 (baseline @ 61.5k) ->
**0.0245** (C @ 92.5k), the search's value error 0.104 / 0.097 -> **0.084**, the value head's
error 0.123 / 0.120 -> **0.115**. The baseline did not move over its last 9k steps, so this is
not just more training. Ladder: C @ 92.5k = 1,928 +- 40, puzzles 76.2%. The 2-hourly
measurements written on C's box were lost with it (outside the synced run dir); the
branch-point numbers above were re-measured on the next box.

Next (2026-10-26 10:35 UTC, budget raised to $215): branch `enc-c800` = C @ 92.5k at
800/100 sims (`configs/enc_sims800_open0_qpol.yaml`), fresh buffer, box runC800 until
~20:30 UTC; its measurements now go to the run's metrics/ (synced home). `src/run.py` now
builds an engine on resume when a branch has none (self-play otherwise waits forever).

### Opening collapse and the 800 branch (2026-10-26)

Self-play openings, latest 3,000 games (scripts/opening_diversity.py, measured on the boxes):
C before the quick-move switch (54k): 1.e4 29%, Nf3 19%, d4 14%, 98% of games distinct by
ply 8. C800 (~104k): **1.e4 96%**, d4 1.6%, Nf3 0.7%; distinct positions 9% at ply 4, 32% at
ply 8, 56% at ply 12. The raw policy already gave 1.e4 ~0.81 at 92.5k (before 800 sims), so the
quick-move policy sampling (T=0.7) did most of the narrowing and 800 sims added to it. The
self-play PGNs of C and C800 were not synced (the sync takes metrics, logs, checkpoints) and
were lost with the boxes; only `~/chessrl-runs/runC-open0/c_sample_100.pgn` remains.

Re-rated on the balanced Stockfish openings (100 sims, `rerate_sf.jsonl` in runA-deep and
runC800 metrics): C 54k 2,205 -> 92.5k 2,427; C800 95k-112k 2,228 / 2,432 / 2,337 / 2,479
(100.5k) / 2,329 / 2,327, i.e. no gain over C at 400 sims on the 100-sim ladder. Against
Stockfish-1400 at 800 sims (first post's test, 100 games from the start position) all six
benchmarked networks score 99-100% (first post's best: 55%); the Stockfish-2000 round runs
overnight (`sf_benchmark.jsonl`).

### Stockfish level at 50% for C800 (2026-10-27, home machine)

Stockfish 16 here = Ubuntu 24.04's package (stockfish_16-1build1, same as the boxes; bench
2593605 nodes), in ~/.local/opt/stockfish16, linked as ~/.local/bin/stockfish. The network
searches on the CPU (scripts/cpu_stockfish_match.py: src/analysis/engine.py, 800 sims, one at a
time) because the home GPU was busy. Check against a box match: sims400 @ 60k vs UCI_Elo 2000,
100 games: 83.5% here (81/5/14) vs 85% on the box (83/4/13). C800 @ 100.5k, 40 games per level,
100 ms per move: 2400 70% (26/4/10), 2700 45% (10/16/14), 3000 5% (0/4/36). One logistic fit:
2,596 +- 39 (2,613 +- 43 from 2400 and 2700 only): **C800 ~ Stockfish 16 UCI_Elo 2,600 at 100
ms per move** on a Ryzen 9 9950X3D. The drop to 5% at 3000 is steeper than a logistic: near
the top of the scale the weakening switches off. Results in artifacts/cpu_sf/.

### The two 24 h baselines (2026-10-27/28)

Side by side, rated live on the balanced-opening ladder (100 sims) with the opening probe; both
moved once (~5.5 h in) with their buffers to Quebec boxes (wd-q EPYC 9554, main-q EPYC Zen 4 ES;
main-q runs the original code on PyTorch 2.14.1+cu126). Training hours, downtime excluded:

| | 1,000 Elo | 2,000 Elo | Elo at 24 h | puzzles at 24 h |
|---|---|---|---|---|
| base-wd (configs/enc_sims400_wd.yaml) | ~2.1 h | 5.6 h | 2,230 | 70% |
| base-main (main@128893c) | 12.0 h | never | 1,170 | 27% |

First post's Stockfish test (800 sims, 100 ms/move, start position, 40 games, network on the CPU):
base-main 15% vs UCI_Elo 1400; base-wd 100% vs 1400 and 79% vs 2000 (31/1/8). With no search
(0 sims) base-wd scores 42.5% vs 1400, 25% vs 1700, 7.5% vs 2000, and solves 58 of the 286
puzzles rated 2,000+. An earlier main run needed 21 h to reach 1,000 on the same scale: the
original code's speed varies a lot between runs.
