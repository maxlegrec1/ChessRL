# Next round: faster, cleaner self-play learning

Started 2026-10-21. Budget: **$200 of rented GPU time** (Vast.ai), tracked by
`scripts/vast/budget_guard.py`. This file is the plan and the lab notebook —
results get appended to the log at the bottom as they land.

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
