# Configuration reference

Every config field, grouped by section as in [`src/train/config.py`](../src/train/config.py).
For each field: the recipe's value (from [`src/train/config.yaml`](../src/train/config.yaml)),
the dataclass default (what a YAML that omits the key gets), and what it does. *unset* means the
recipe does not set the key, so the default applies; *run path* means the launcher rewrites it to
a path inside the run directory. [`ARCHITECTURE.md`](ARCHITECTURE.md) explains the mechanisms.

## How configs load

- `python -m src.run` reads `src/train/config.yaml` for a new run, or the file given with
  `--config`. `--resume <run_dir>` reads `<run_dir>/config.yaml`, unless `--config` is given.
- `load_yaml` resolves `extends: <path>` (relative to the file that contains it), recursively:
  the parent is loaded first and the child wins. Sections merge key by key; scalars and lists are
  replaced. A variant can therefore start with `extends:` pointing at `src/train/config.yaml` and
  list only what changes.
- The launcher writes the merged YAML to `<run>/config.yaml` after rewriting the paths
  (`data.dataset_path`, `checkpoint.dir`, `checkpoint.engine_dir`, `checkpoint.iteration_file`,
  `selfplay.pgn_dir`, `wandb.dir`) and, on resume, setting `resume_path` and keeping any earlier
  `wandb.resume_id`. Self-play and the trainer both read that snapshot. It holds the keys the YAML
  sets and the rewritten paths, without `extends:` and without the dataclass defaults.
- `load_config` starts from the dataclass defaults and overlays the YAML. **Unknown keys are
  ignored without a warning**, so a misspelt key silently keeps the default.
- **The dataclass defaults are not the recipe.** Optional features default to off, and several
  basic values differ (`batch_size` 256, `learning_rate` 1e-5, `optimizer: adam`,
  `selfplay.simulations` 800, `selfplay.max_buffer_size` 10,000,000, among others). A config
  should set every value it relies on. The recipe does; the keys it leaves out are optional
  features that stay off (growth, the cosine schedule, resign calibration, `torch.compile`) and
  logging details. Since snapshots store only the YAML's keys, changing a default in `config.py`
  also changes it for resumed runs whose YAML omits that key.
- `python -m src.self_play --config <yaml>` and `python -m src.train --config <yaml>` also run on
  their own (default `src/train/config.yaml`). Their paths are then used as written.

Launcher flags (`python -m src.run`):

| Flag | Effect |
|---|---|
| `--config PATH` | Base YAML. Default: the recipe for a new run, the run's own `config.yaml` on `--resume`. |
| `--name NAME` | Suffix of the new run's id (`<date>_<time>_<4 hex>_<name>`). |
| `--resume RUN_DIR` | Resume that run from its newest `checkpoints/step_*.pt`. |
| `--no-train`, `--no-selfplay` | Start one side only (attached to the terminal). |
| `--no-foreground` | With one side, write its output to the log file instead. |
| `--shutdown-grace S` | Seconds to wait for the children to exit before SIGKILL (15). |

## Top level (`TrainingConfig`)

| Key | Recipe | Default | What it does |
|---|---|---|---|
| `device` | `cuda` | `cuda` | Torch device for training. Self-play always runs on the GPU (the fused CUDA network). |
| `batch_size` | 2048 | 256 | Records per training step. |
| `learning_rate` | 0.001 | 1e-5 | Peak learning rate; `lr_schedule` and growth re-warmup scale it. |
| `optimizer` | `adamw` | `adam` | `adamw`: `torch.optim.AdamW`, decoupled weight decay. `adam`: `torch.optim.Adam`, where `weight_decay` becomes an L2 term in the gradient. |
| `weight_decay` | 0.05 | 0.0 | Applied only to parameters with two or more dimensions (weight matrices, embeddings, the attention pair bias); norm gains, biases and scalar gains are exempt. Decoupled only with `optimizer: adamw`. The recipe uses it to keep the encoder's activations inside FP16 range. |
| `beta1`, `beta2` | 0.9, 0.999 | 0.9, 0.999 | Adam betas. |
| `grad_clip` | 1.0 | 0.0 | Maximum global gradient norm, applied after unscaling the FP16 loss scale. 0 = off. |
| `amp_dtype` | `float16` | `float16` | Autocast dtype: `float16` (also `fp16`, `half`) with a gradient scaler, or `bfloat16` (`bf16`) without one. Unknown values fall back to float16. |
| `log_every` | 5 | 50 | Steps between rows of `metrics/train.jsonl` (and W&B). The console prints every 10 rows. |
| `policy_target` | `visits` | `played` | `visits`: cross-entropy to the record's visit distribution. `played`: cross-entropy to the move played. Both losses are logged either way. The default `loss_weights.policy` (0.1) is the one for `played`; the recipe pairs `visits` with 1.0. |
| `search_heads` | `outcome` | `search` | Which heads the engine gives MCTS. `outcome`: value and mlh, trained on game results. `search`: value_q and mlh_q, trained on the search's own estimates. `q_value` (value_q with mlh) and `blend` (the average of both WDLs, with mlh) are value-only variants. Applied at export from the config saved in each checkpoint, so a change affects networks exported from later checkpoints; `src.export.to_cnet --search_heads` overrides it. |
| `mirror_prob` | 0.5 | 0.5 | Probability that a sampled record is colour-flipped (ranks mirrored, colours swapped, moves remapped). The only augmentation. |
| `compile` | true | false | Wrap the model in `torch.compile` for training: the step takes 34 ms instead of 102 on an RTX 4090 (batch 2,048, the 128 x 4 encoder); the first step compiles, about a minute, and growing the network compiles again. |
| `mps` | true | false | `src.run` starts an NVIDIA MPS daemon for the run when it launches both processes, so self-play's and the trainer's kernels share the GPU instead of taking turns, and gives the trainer below-normal priority (`CUDA_MPS_CLIENT_PRIORITY=1`). If the daemon or a test client fails, the run goes on without MPS. Self-play alone or the trainer alone never uses it. |
| `async_engine_build` | true | false | Build engines in a background process while training continues. A checkpoint saved while the previous build still runs gets no engine (`build_skipped` event). false: the trainer waits for each build. |
| `metrics_path` | *unset* | null | Trainer metrics file. null: `<run>/metrics/train.jsonl` (the parent of `checkpoint.dir`). Self-play always writes `<run>/metrics/selfplay.jsonl`. |
| `expensive_metrics_every` | *unset* | 50 | Steps between host-syncing diagnostics (`legal_move_prob`, the policy mass on legal moves, from 16 rows with python-chess). They are computed on the step that ends at a multiple of this value, so they appear in that step's row when `log_every` divides it (the recipe's 5 and 50). |
| `resume_path` | set by `--resume` | null | Checkpoint to resume from. |
| `run_name` | *unset* | null | W&B run name when `wandb.run_name` is unset. |

## `model`

The architecture of a new network (the launcher's initial engine and a fresh trainer). A resumed
trainer rebuilds the architecture stored in its checkpoint (`model_spec`) instead.

| Key | Recipe | Default | What it does |
|---|---|---|---|
| `arch` | `encoder` | `resnet` | `encoder`: the transformer in `src/models/encoder.py`. `resnet`: `src/models/small/small_model.py`. |
| `d_model` | 128 | 128 | Encoder width. |
| `layers` | 4 | 4 | Encoder blocks. |
| `heads` | 8 | 8 | Attention heads (`d_model` must divide by it). |
| `d_ff` | 256 | 256 | Hidden width of the SwiGLU feed-forward. |
| `blocks` | *unset* | 4 | ResNet residual blocks. |
| `channels` | *unset* | 64 | ResNet trunk width. |

The recipe's encoder has 812,557 parameters; the default 4 x 64 ResNet has 1,285,921.

## `lr_schedule`

The learning rate is `learning_rate` x schedule factor x warmup factor (and the growth re-warmup).

| Key | Recipe | Default | What it does |
|---|---|---|---|
| `kind` | `constant` | `constant` | `constant` or `cosine`. |
| `warmup_steps` | 1000 | 0 | Linear warmup from `learning_rate / warmup_steps` to `learning_rate`, counted from `start_step`. |
| `start_step` | *unset* | 0 | Step the warmup and the cosine count from. -1: the step the run resumes at, resolved once and saved in checkpoints (`lr_start_step`) so later resumes keep it. |
| `total_steps` | *unset* | 0 | Cosine only: length of the schedule from `start_step`, including `hold_steps`. 0 disables the cosine. |
| `hold_steps` | *unset* | 0 | Cosine only: steps at the full rate before the decay starts. |
| `final_ratio` | *unset* | 0.1 | Cosine only: the final rate as a fraction of `learning_rate`. |

## `loss_weights`

The total loss is the weighted sum of the five head losses. Moves-left targets are plies / 25,
capped at 4 (100 plies).

| Key | Recipe | Default | What it does |
|---|---|---|---|
| `policy` | 1.0 | 0.1 | Policy cross-entropy (target set by `policy_target`). |
| `value` | 1.0 | 1.0 | Value head: cross-entropy to the game result (one-hot WDL from the side to move). |
| `value_q` | 1.0 | 1.0 | value_q head: cross-entropy to the root WDL after the search. |
| `mlh` | 0.05 | 0.05 | mlh head: squared error to the plies remaining in the game. |
| `mlh_q` | 0.05 | 0.05 | mlh_q head: squared error to the search's backed-up plies at the root. |

## `checkpoint`

| Key | Recipe | Default | What it does |
|---|---|---|---|
| `dir` | *run path* `checkpoints/` | `artifacts/checkpoints` | Where `step_NNNNNNNN.pt` go. Its parent counts as the run directory (metrics, `engine_builds.log`, the ladder file read by growth). |
| `every_steps` | 500 | 1000 | Training steps between checkpoints, and so between network exports. |
| `keep_last` | 999999 | 5 | After each save the trainer deletes all but the newest `keep_last` `step_*.pt`. 0 or less keeps all. |
| `build_script` | `scripts/build_net.sh` | `scripts/build_net.sh` | Run after each checkpoint as `<script> <ckpt> <step> <engine_dir> <iteration_file>`; null = no networks. The launcher's initial and resume networks always use `scripts/build_net.sh`, and a config naming the old `scripts/build_trt.sh` runs `build_net.sh`. |
| `engine_dir` | *run path* `engines/` | `artifacts/engines` | Where networks (`.cnet`) are exported. |
| `iteration_file` | *run path* `engines/.iteration` | `artifacts/engines/.iteration` | CSV (`iteration,engine_path`) the build script appends to and self-play reads. |

## `data`

| Key | Recipe | Default | What it does |
|---|---|---|---|
| `dataset_path` | *run path* `buffer/buffer.bin` | `artifacts/buffers/buffer.bin` | The buffer file; its sidecars `.writes` and `.stats.json` sit next to it. Self-play writes it, the trainer memory-maps it. |
| `min_samples_to_start` | 50000 | 8192 | Records the buffer file must hold before the first training step. |
| `warmup_poll_seconds` | 5 | 10.0 | Poll interval while waiting for them (at least 1 s). |
| `target_k` | 5.0 | 5.0 | Pacing: the trainer trains on at most `target_k` samples per record written since it started (minimum 0.1), and sleeps when ahead. |
| `window_c` | 250000 | 0 | Scale c of the sampling window over the newest records, W(N) = c (1 + β ((N/c)^α - 1) / α) for N records ever written (W = N up to c), capped by the buffer. 0 or less samples the whole buffer. **Needs `selfplay.fifo_buffer: true`** (read from the buffer's `.stats.json`); otherwise ignored. |
| `window_alpha` | 0.75 | 0.75 | Exponent α of the window's growth. |
| `window_beta` | 0.4 | 0.4 | Rate β of the window's growth. |

With the recipe's values the window holds 494k records at N = 1M, 751k at 2M, 1.38M at 5M, and
reaches the whole 2M buffer at about N = 8.5M.

## `selfplay`

`boards`, `poll_seconds`, `pgn_dir`, `resign_target_fp` and `resign_calibration_window` are used
by the Python side (`src/self_play/__main__.py`); every other field is copied by name onto the
C++ `SelfPlayConfig` (`_CPP_FIELDS`). A searched ply is one played by MCTS, not a random opening
ply.

| Key | Recipe | Default | What it does |
|---|---|---|---|
| `boards` | 1024 | 1024 | Concurrent games. Each worker step searches one move in every game. |
| `threads` | 128 | 256 | CPU threads running the searches. Also the inference batch size at which a batch is dispatched. |
| `async_play` | true | false | Each search thread plays its own boards in turn, with no step barrier: the GPU never waits for a step's last searches or for the moves to be applied. A worker step then returns after one move per board's worth of moves. false: each step searches every board once, all at once. |
| `hash_size` | 0 | 0 | Transposition table size in MB, shared by all searches. 0 = off. |
| `simulations` | 400 | 800 | Simulations of a full search. |
| `pcr_full_prob` | 0.25 | 1.0 | Playout-cap randomisation: probability that a move gets a full search, which is recorded and gets Dirichlet noise. 1.0 = every move. |
| `pcr_fast_simulations` | 50 | 100 | Simulations of the other moves (not recorded, no noise). |
| `tree_reuse` | true | false | Keep the subtree of the move played for the game's next search, which continues it: `simulations` (or `pcr_fast_simulations`) is then the number of visits the root's moves end up with, and only the missing ones are run. A full search mixes fresh Dirichlet noise into a reused root. Not for policy-sampled moves. |
| `quick_policy_temperature` | *unset* | 0.0 | Optional. When > 0, fast moves during the sampling phase (`temperature_plies`, root temperature > 0) skip the search and sample the raw policy at this temperature (one network call). 0 = off. |
| `cpuct` | 0.65 | 0.65 | PUCT exploration constant. |
| `temperature` | 1.3 | 1.3 | Prior temperature: priors are softmax(logits / `temperature`) at every node. 0 puts the whole prior on the top move. |
| `temperature_at_root` | 1.0 | 1.0 | Move choice during the first `temperature_plies`: sampled with probability proportional to visits^(1 / T). 0 = the most visited move. |
| `temperature_plies` | 30 | -1 | Searched plies that use `temperature_at_root`. -1 = the whole game. |
| `temperature_late` | 0.0 | 0.0 | Root temperature after `temperature_plies`. 0 = the most visited move. |
| `virtual_loss_width` | 8 | 8 | Leaves each search collects with virtual loss before one network round. 1 = strictly sequential search. |
| `use_dirichlet` | true | true | Root Dirichlet noise on full searches (never on fast ones). |
| `dirichlet_alpha` | 0.3 | 0.12 | Dirichlet α. |
| `dirichlet_epsilon` | 0.25 | 0.05 | Noise weight ε: root prior = (1 - ε) P + ε noise. |
| `random_opening_plies` | 8 | 0 | Uniformly random legal plies at the start of every game. They produce no records. |
| `max_plies` | 600 | 600 | Searched plies after which a game is a draw (flag 32 in its records). |
| `resign_threshold` | 0.05 | 0.0 | Root expected score (W + D/2) below which the side to move counts towards resigning. 0 = never resign, and no playthrough games. |
| `resign_consecutive` | 3 | 2 | Own moves in a row below the threshold before resigning. Moves sampled from the raw policy (`quick_policy_temperature`) neither extend nor reset the count. |
| `resign_playthrough` | 0.1 | 0.1 | Fraction of games that never resign; in them a would-be resignation is counted, and counted as wrong if that side did not lose. |
| `resign_target_fp` | *unset* | 0.0 | Optional calibration. When > 0 (with `resign_threshold` > 0): after every `resign_calibration_window` would-be resignations in playthrough games, the threshold is multiplied by 0.8 if the wrong share exceeded this target, or by 1.1 if it was under half of it, within 0.005 to 0.2. 0 = fixed threshold. |
| `resign_calibration_window` | *unset* | 400 | Would-be resignations per calibration step (`resign_target_fp` only). |
| `max_buffer_size` | 2000000 | 10000000 | Buffer capacity in records (1,280 bytes each: 2.56 GB for the recipe). |
| `fifo_buffer` | true | false | When the buffer is full: true = ring, the oldest record is overwritten; false = a random record is replaced. `data.window_c` needs true. |
| `poll_seconds` | 300 | 300 | How often self-play checks `.iteration` for a new engine. |
| `save_pgn` | false | false | Write each finished game as a PGN file in `pgn_dir`. |
| `pgn_dir` | *run path* `pgns/` | `artifacts/pgns` | Directory for those PGNs. |

The C++ `SelfPlayConfig` also has `curriculum_prob` and `curriculum_max_pieces` (games from random
few-piece positions); the Python config does not expose them, so they stay at 0 and 4.

## `growth` (optional; off in the recipe)

Function-preserving trunk growth during training (`src/models/growth.py`). None of these keys is
in the recipe.

| Key | Recipe | Default | What it does |
|---|---|---|---|
| `schedule` | *unset* | `[]` | Stages `[step, layers]` for the encoder or `[step, blocks, channels]` for the ResNet. Once the trainer reaches `step`, the trunk grows to that size (never shrinks) without changing the network's output. Empty = no growth. |
| `trigger` | *unset* | `steps` | `steps`: grow when a stage's step is reached. `plateau`: also wait until the run's ladder Elo (`<run>/ladder/ladder.jsonl`) has flattened; checked every 100 steps. |
| `lr_rewarmup_steps` | *unset* | 200 | After a growth the optimizer is rebuilt (its state starts fresh) and the learning rate ramps back up linearly over this many steps. |
| `plateau_hours` | *unset* | 2.0 | `plateau` only: the ratings since the last growth must span this long, and the gain is measured over the last `plateau_hours`. |
| `plateau_min_gain` | *unset* | 50.0 | `plateau` only: grow when the mean Elo of the window's last third minus its first third is below this. |
| `plateau_min_ratings` | *unset* | 4 | `plateau` only: ratings needed since the last growth. |

## `wandb`

Metrics always go to `<run>/metrics/*.jsonl`, which the dashboard reads. W&B is an optional copy
of the trainer's rows.

| Key | Recipe | Default | What it does |
|---|---|---|---|
| `enabled` | false | true | Also log the trainer's numeric metrics to Weights & Biases. |
| `project` | `self_play` | `self_play` | W&B project. |
| `entity` | null | null | W&B entity (user or team). |
| `run_name` | *unset* | null | W&B run name; falls back to the top-level `run_name`. |
| `dir` | *run path* `wandb/` | null | W&B local directory. |
| `resume_id` | *unset* | null | Attach to an existing W&B run id instead of starting one. `--resume` keeps it from the run's previous config. The trainer writes the id it uses to `<run>/wandb_run_id.txt`. |

## Environment variables

Settings outside the YAML, read by the C++ extension at run time or by the build and tools.

| Variable | Default | What it does |
|---|---|---|
| `CHESSRL_INFER_WORKERS` | 6 | Inference worker threads per batch runner, each with its own CUDA stream. |
| `CHESSRL_KERNEL_BLOCKS` | 2 | 2: the fused kernel keeps two blocks (positions) on each SM, at most 128 registers per thread; 1: the one-block build (up to 255 registers). Only with the default FP16 accumulation; the other accumulation settings always use one block. |
| `CHESSRL_MCTS_FLUSH_US` | 100 | Microseconds the batcher waits for a full batch before dispatching a partial one. |
| `CHESSRL_FP16_ACC` | 256 | The fused network's projection multiplies sum in FP16: N terms per FP16 sum before an FP32 addition (32, 64, or 256 for whole sums); 0 keeps FP32 accumulators. The attention and policy-logit multiplies and the residual stream are FP32 either way. |
| `CHESSRL_BLOCKING_SYNC` | 0 | 1: inference workers sleep on a blocking-sync event while the GPU runs their batch instead of spinning (saves about 2 cores, costs 3 to 4% when the CPU is not short). |
| `CHESSRL_SEARCHES_PER_THREAD` | 1 | Searches each self-play thread interleaves (`threads` x this many in flight; the batch threshold counts them all). Fewer threads wake less often and use less CPU, but each search waits for its thread; 32 x 4 was 2% slower than 128 x 1 on the 4090 box (`docs/PERF.md`). |
| `CHESSRL_MCTS_PHASES` | unset | Set: print where the search threads' time goes after each self-play step (`[mcts-phases]`) and the step's serial parts (`[step-phases]`). |
| `CHESSRL_MCTS_SMALL_BATCH_SIZE`, `CHESSRL_MCTS_BATCH_SIZE` | 1024 | Maximum batch of searches outside self-play (ladder, scripts); the first takes precedence. |
| `CHESSRL_CUDA_ARCH`, `CUDA_ROOT` | `sm_90`, `/usr/local/cuda` | Build time (`uv sync`): GPU architecture (the fused kernel needs `sm_80` or newer) and toolkit location. |
| `CHESSRL_RUNS_ROOT` | `artifacts/runs` (dashboard), `~/chessrl-runs` (analysis board) | Where the dashboard and the analysis board look for runs; colon-separated for the dashboard. |
