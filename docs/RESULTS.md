# What we learned

The project in three rounds: what each one changed and what moved the strength. Ratings are on the
project's ladder (random play = 0, balanced openings chosen by Stockfish, about ±45 Elo per
rating) unless said otherwise. Most results below come from single runs, so differences under about
50 Elo are within the noise. The detailed lab notes behind them are not kept in the repository.

## Round 1: the first version

A 1.29M-parameter ResNet, trained for two weeks on a single RTX 4090 (and a second, instrumented run
of about ten days on a rented H200). It reached roughly the level of Stockfish 16 limited to 1400 Elo
at 100 ms per move, and scored 55% against it at best. The bugs taught the most:

- **A mirror augmentation that flipped moves wrongly** taught half the data wrong labels. Training
  still progressed, slowly. A legal-move probability metric showed it all along.
- **Castling starvation**: Dirichlet noise set far below AlphaZero's values meant the search never
  tried castling, so the policy never learned it. Back to α 0.3, ε 0.25.
- **Empty history planes at inference**: self-play fed the network no history while the trainer
  used it. Fixed with parity tests between the Python and C++ encoders.

Written up in [the first post](https://mvdbk.fr/blog/chessrl-alphazero-single-gpu/).

## Round 2: six times faster

Same GPU, same 24 hours: the original code reached 1,170, the new recipe 2,230 (1,000 Elo after 2.1 h
instead of 11.8 h, 70% of the puzzles instead of 27%). What mattered:

| Change | Effect |
|---|---|
| Policy learns the search's visit counts, not the move played | 500-750 Elo |
| Search reads the value head trained on game results, not the one trained on its own estimates | faster early training |
| Transformer (one token per square, 0.81M parameters) instead of the ResNet | +370-590 Elo at equal positions |
| 400/50 simulations with playout-cap randomisation (25% full searches) | twice the games per hour |
| Weight decay 0.05 | kept activations inside FP16 range |

An FP16 bug looked like a training collapse: RMSNorm squares overflowed in the 16-bit engine once
activations passed 256, so self-play played with a different network than the one being trained.
Written up in [the second post](https://mvdbk.fr/blog/alphazero-from-scratch-rtx-4090/).

## Round 3: speed, then the plateau

**Speed.** A fused CUDA kernel runs the whole network in one launch per batch and replaced TensorRT.
With tree reuse, asynchronous play and other changes, self-play records **8.6 times** as many
positions per second as the round-2 recipe on the same box. The network then went to 8-bit
integers (`model.int8`: the kernel's projections in INT8, 2.9M positions/s against 2.24M in FP16,
and training with the same rounding), and the search lost its per-search allocations and system
calls, with one thread watching the GPU's batches: **11 times** the round-2 recipe self-playing
alone (about 5,650 recorded positions/s), 4,800 against 3,980 in the 8-hour runs, for the same
rating at 8 hours. The trainer step went from 26.8 ms to
11.2 ms (Triton fused kernels and an FP16 residual stream). Positions are now
canonical end to end: the network always plays white, with one 1,858-move vocabulary.

**Muon against AdamW.** In this setting Elo followed how much the policy moved between checkpoints.
Muon needed two to four times its usual learning rate to move the policy as much as AdamW, and then
only matched it. Set aside, still available as `optimizer: muon`. With INT8 it did worse: about 50
Elo below AdamW per trainer step at 250,000 steps (2,000 games head to head), more than in FP16.

**Resuming into an empty buffer costs about an hour.** A run restarted from its last checkpoint with
a new buffer trained on a window of only 170k to 800k records. The value head memorised those games,
became overconfident, resigned too early, and lost 150 to 400 Elo (puzzles 0.73 to 0.66) until the
window refilled. Branch runs now start with a copy of their parent's buffer.

**At the plateau** (2-hour branches from a checkpoint at about 2,300, each with one change; control
2,300 ± 14):

| Change | Δ Elo |
|---|---|
| Sample the move for 12 searched plies instead of 30 | +77 |
| Fast moves drawn from the raw policy at temperature 0.7 | +73 |
| Learning rate decayed to 0 | +25 (puzzles +0.04) |
| No random opening plies | +3 |
| Fresher networks for self-play (poll every 60 s) | −52 |
| Resignation calibrated to 3% false positives | −66 |
| Search reads the value trained on its own estimates | −169 |

The last one spirals: the search-trained value grows overconfident, games end early by resignation,
draws vanish, and the targets feed the overconfidence. It failed the same way in a run from scratch.

**From scratch, 4 hours** (mean Elo over hours 2-4):

| Run | Change from the previous recipe | Elo |
|---|---|---|
| Previous recipe | (8 random plies, 30 sampled, 2M buffer, constant LR) | ~2,215 |
| win6m | 6M-record buffer | 2,317 |
| t07 | 6M, fast moves sampled at temperature 0.7 | 2,326 |
| open40t07 | 6M, fast moves at 0.7, no random plies, 40 sampled plies | 2,327 |
| combo | 6M, 12 sampled plies | 2,348 |
| combolr | combo, learning rate halved at step 70k | 2,391 |

A 2M buffer stopped the sampling window at about 8 minutes of self-play; 6M lets it keep growing. The
fast 50-simulation searches have flat visit counts, so sampling them at temperature 1 plays many
weak moves; sharpening them to 0.7 is what made openings sampled from the initial position work
without collapsing onto one first move.

**The current recipe** is open40t07 with the learning rate flat for 2 hours, then a cosine down to a
tenth by 8 hours (open40cos):

| Hours | 2-2.5 | 3-3.5 | 4-4.5 | 5-5.5 | 6-6.5 | 7-7.5 | 7.5-8 |
|---|---|---|---|---|---|---|---|
| Elo | 2,242 | 2,325 | 2,395 | 2,365 | 2,451 | 2,515 | **2,525** |
| Puzzles | 0.71 | 0.74 | 0.76 | 0.77 | 0.78 | 0.80 | **0.81** |

Every constant-learning-rate run levelled off around 2,300-2,400; the decay kept this one climbing.
Its openings stayed varied: at the end, 6 effective first moves (the most common one in a third of
the games) and about 4,400 distinct positions at ply 8 in 400,000 records.

This is one 8-hour run. It does not separate the schedule's share of the gain from the openings' and
the buffer's; the matching control is the previous recipe with the same schedule.
