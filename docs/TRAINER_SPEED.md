# Trainer speed: what to do, in order

Not urgent today, but the trainer's step is the next thing that limits a run. This note lists
the ways to make it cheaper, ranked by how much **GPU time** each one removes from a step, since
on a shared GPU every second the trainer spends there is a second self-play waits. Ideas taken
from the supervised stack in `~/chess_speedrun` (`speedrun/IMPROVEMENTS.md`, `speedrun/IDEAS.md`)
are marked with their measured effect there.

## Where things stand (2026-10-30)

Short real runs on an RTX 4090 with a Ryzen 9 7950X (run C's checkpoint, the recipe with the
fused network and tree reuse, 12 minutes each, the last 8 measured; `docs/PERF.md`):

| setup | records/s | trainer steps/s | step time | samples per record | trainer waiting | GPU busy |
|---|---|---|---|---|---|---|
| self-play alone | 2,006 | | | | | 79% |
| trainer + self-play | 1,327 | 3.24 | 0.14 s | 5.0 | 52% | 89% |
| both under MPS | 1,662 | 4.06 | 0.22 s | 5.0 | 9% | 98% |
| MPS, trainer at below-normal priority | 1,964 | 3.46 | 0.28 s | 3.65 | 0% | 99% |
| same, through the launcher, FP16 accumulation in self-play (today's recipe) | 1,994 | 4.86 | 0.17 s | 5.0 | 16% | 93% |
| + zero-copy hand-back in self-play | 2,347 | 4.65 | 0.21 s | 4.1 | 0% | 98% |
| + `torch.compile` in the trainer (today's recipe) | 2,402 | 5.85 | 0.076 s | 5.0 | 50% | 86% |

Without MPS the trainer keeps up (it reaches the recipe's 5 samples per record, `data.target_k`,
and waits half the time), but the two processes take turns on the GPU and self-play makes a
third fewer records. Under MPS their kernels share the GPU; with the trainer at low priority,
self-play loses only 2% and the trainer gets what is left, 3.46 steps/s, not enough for 5
samples per record at 1,964 records/s (that needs 4.8). From there every GPU-second taken off a
step is more training at no cost to self-play. FP16 accumulation in the self-play kernel freed
enough: the recipe reached 5 samples per record with the trainer waiting 16% of the time. Then the
zero-copy hand-back made self-play 18% faster (last row), and the trainer is the limit again at
4.1 samples per record: it needs 5.7 steps/s to keep up. The items below are how to get there.

The step is not compute-bound. The network is small (0.81M parameters, about 0.6 TFLOP per
step of 2,048 positions, forward and backward), and at 0.14 s (sharing the GPU in turns) that
is about 4 TFLOP/s on a
card that does 165 with FP32 accumulation. The speedrun trains a 30 times larger model at
12,600 positions/s on the same card. The time goes to memory traffic (eager attention and
element-wise kernels), kernel launches and host syncs.

## Measured (2026-10-30)

`scripts/bench_trainer.py` (the real Trainer, resumed from run C's checkpoint, sampling a real
self-play buffer) on the RTX 4090 alone, batch 2,048:

| | eager (the recipe) | `torch.compile` |
|---|---|---|
| step, one batch reused | 102.4 ms | 33.5 ms |
| step, fresh batches (as `run()` does) | 102.7 ms | 33.9 ms |
| kernels per step | 1,264 | 687 |

The data pipeline is not a limit (the Prefetcher makes 530 to 580 batches/s and fresh batches
cost nothing over a reused one), and the step is not launch-bound: the GPU is busy the whole
102 ms. Where it goes, eager (the PyTorch operators that launched the kernels): the residual
stream's RMSNorms, forward and backward, about 30%; the attention core on the [2,048, 8, 64, 64]
scores about 25% (FP16/FP32 casts of the scores 5 ms, softmax and its backward 6 ms, bias add and
its gradient 3 ms, the batched matmuls 5 ms); the QK-norm about 20%; the matrix multiplies of all
linear layers about 5%. Memory-bound element-wise work, which compiling fuses: 3 times faster.
What is left on top after compiling is mostly attention (its fused softmax kernels and batched
matmuls), the next item's target.

Compiled against eager on one real batch with run C's weights (`torch.autocast` FP16, as the
trainer runs): outputs within FP16 rounding (policy logits 0.15 at a scale of 26), the loss equal
to 6 digits, gradients within 3% per tensor (the QK-norm gains, the pair bias and the x0 gains,
small reduction-heavy gradients; batch-to-batch noise is far larger). In the real run (last row of
the first table) the compiled trainer keeps up with self-play at full speed: 5 samples per record,
waiting half the time. `compile: true` is in the recipe since then.

**Fused attention** (`F.scaled_dot_product_attention` with the pair bias as its additive mask,
PyTorch's memory-efficient kernel): the step takes 80.1 ms eager and **26.4 ms compiled** (from
102.4 and 33.5), 3.9 times faster than the eager step with the explicit attention. Against the
explicit formula in the model's layout: outputs within FP16 rounding, gradients of q, k, v and the
pair bias within 0.06% (FP32 evaluation within 3e-6); the fused self-play kernel's parity is
unchanged. What is left of the compiled step is spread thin: the attention backward about 2 ms,
the fused SwiGLU, norm and QK-norm kernels 0.6 to 1.5 ms each, the matrix multiplies about 5 ms
in all. The items below would each win a few percent; the trainer already waits for records half
the time in a real run.

## Ranked by GPU time saved

1. **Fused attention.** `src/models/encoder.py` builds the full score tensor in every layer:
   [2,048, 8, 64, 64] scores (67M values, 134 MB in FP16), then the bias add, the softmax and
   their backward, each a pass over that tensor. Activations are only [2,048, 64, 128] (17M). A
   fused kernel that never writes the scores removes most of the step's memory traffic.
   - Cheapest: `F.scaled_dot_product_attention` with the pair bias as a contiguous FP16 mask.
     In the speedrun that alone put SDPA on its fused kernel: +19% step.
   - Better: the speedrun's Triton kernel (`speedrun/attn.py`, one program per position and
     head, S = 64 and D = 16 supported), which also sums the shared bias gradient per head
     instead of writing it per position. -11.5% step at width 512 there; likely more here,
     where attention is a bigger share. Our QK-norm has learnable gains: add them to the kernel
     (or drop them, which needs a training A/B).
2. **`torch.compile` of the forward and the losses** (`compile: false` today), with fixed batch
   shapes (`dynamic=False`). Fuses the element-wise work: norms, SwiGLU, residual adds, x0
   re-injection, the losses. Memory-bound kernels were 45% of the speedrun's step even at width
   512. Part of a x1.34 bundle there (with the next item).
3. **Fewer per-step metrics and host syncs.** About 28 metric reductions run every step
   (`trainer.py` `_train_step`), `bool(mirror.any())` and `scaler.get_scale()` sync with the host
   every step. Accumulate the metrics on the GPU and read them every 100 steps; remove the
   syncs. Saves GPU kernels and, more, the gaps between them.
4. **FP16 accumulation in the matrix multiplies**
   (`torch.backends.cuda.matmul.allow_fp16_accumulation = True`). The 4090's tensor cores run
   twice as fast with FP16 accumulators (330 against 165 TFLOP/s on paper; 209 against 154
   measured on a large GEMM). Speedrun: +14% step GPU-only, +8.6% in training, identical loss
   (1.3359 against 1.3358). Our multiplies are small (K = 128 or 256), so the share they take is
   lower, but the risk is low too. It also matches the self-play kernel if that moves to FP16
   accumulation.
   - Keep the residual stream in FP32 or FP16 as today (autocast), and keep the norms, softmax,
     logits and losses in FP32. The speedrun's FP16 residual stream gave the same held-out loss
     but about 20 Elo less in play.
   - Weight decay (0.05 on the matrices, already in the recipe) keeps weights and activations
     small, which is what FP16 accumulators need.
5. **A plain RMSNorm in FP32.** Ours divides each row by its maximum before squaring (two extra
   reductions per norm, 17 norms per forward) to survive FP16. Computing the norm in FP32 (as
   the speedrun and the fused self-play kernel do) is overflow-safe without it. Becomes free once
   item 2 fuses the norm.
6. **Fused AdamW** (`optim.AdamW(..., fused=True)`): one kernel for all parameters instead of
   several per tensor. Small at 0.81M parameters, but free.

## Scheduling: let self-play go first

The GPU is idle part of the time even next to the trainer (self-play's batches leave gaps).
Training kernels in those gaps cost self-play nothing.
- **MPS with a lower client priority for the trainer** (`CUDA_MPS_CLIENT_PRIORITY=1`; the
  driver on the rented boxes supports it). Measured above: self-play 1,964 records/s instead of
  1,327 (MPS without the priority: 1,662). Needs the launcher to start the MPS daemon and to
  give the trainer the variable (the measurement started both by hand).
- **One process, two stream priorities**: self-play's streams at high priority, the trainer's at
  low, so the GPU schedules self-play's blocks first. Needs the trainer inside the self-play
  process (or self-play inside the trainer's): a bigger change, worth it only if MPS falls short.

## Lower GPU payoff, still useful

7. **Replay window on the GPU** (the speedrun's `gpu_data.py` pattern). Today a batch is read
   from the memmap, decoded in NumPy and mirrored in Python (`src/train/dataset.py`), on a CPU
   shared with 128 search threads. The sampled window (250k records and up, x 1,280 bytes, about
   320 MB) fits on the GPU: copy new records in as they arrive and sample on the GPU. +14% there;
   here mostly frees CPU for self-play.
8. **CUDA graphs for the whole step.** The speedrun did not need them (its GPU was 97-99% busy),
   but a 0.81M model is launch-bound. Needs static batch buffers, the learning rate as a GPU
   tensor and no host sync in the step (item 3), and a loss scaler that does not sync (or BF16,
   which needs none but would no longer match the FP16 self-play kernel).
9. **Batch 1,024 instead of 2,048** (x1.11 per position in the speedrun's supervised training).
   Twice the steps at the same `target_k`, so only after items 2, 3 and 8 cut the per-step cost.

## Not worth it at this size

- INT8 or FP8 training multiplies: they pay only where the multiplies are tensor-bound, from
  width 512 up (INT8 everywhere: +5.9% at width 512 in the speedrun; none on its attention
  projections; FP8 no gain at 512).
- INT8 weight gradients: slower even at the speedrun's largest size (-6.7%).
- Muon's precision and compilation tricks: the optimizer is a few percent of a step at 25M
  parameters, negligible at 0.81M. (Whether Muon beats AdamW here at all would need a training
  A/B; the speedrun never compared them.)
- An FP16 residual stream: +4.9% in the speedrun, about -20 Elo in play.
