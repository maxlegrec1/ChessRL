# archive/legacy_scripts/

Frozen scripts kept for reference only. **Nothing in `scripts/` or `src/`
should import or invoke these.** If you find a live reference, delete it.

These are kept (rather than deleted) because each captures a real
investigation or one-time migration that may be useful to re-read when
debugging analogous issues.

## Contents

### Buffer migrations (one-time, run once and done)

- `migrate_buffer_backed_plies.py` — added `plies_remaining_backed`
  column to existing self-play buffers when MLH was introduced.
- `migrate_buffer_mlh_sentinel.py` — stamped legacy rows with
  `plies_remaining = -1` so the trainer's MLH mask treats them as
  unlabeled.

### MLH / value-head investigations (May 2026)

- `probe_halfmove_clock.py` — swept halfmove-clock on identical board
  positions; exposed the value head's 50-move-rule sensitivity.
- `diagnose_mate_blindness.py` — replayed brawl PGNs, compared
  step_9900 policy vs random on mate-in-1 positions.
- `mlh_inspect.py` — buffer MLH-coverage audit; checks how many rows
  have the sentinel vs a real plies-to-end target.
- `sweep_trajectory.py` — cached per-checkpoint `learning_check.py`
  metrics across a full run; fed `full_trajectory_plot.py`.
- `full_trajectory_plot.py` — six-panel matplotlib summary of the above.

### PGN analyzers (May 2026, used to inspect early brawls)

- `analyze_pgn.py`, `analyze_pgn_generic.py` — re-ran MCTS-600 on every
  ply of a PGN; useful for "where did the model blunder" post-mortems.
- `annotate_pgn.py` — annotated PGNs with W/D/L probabilities + move
  quality comments.

### Older legacy probes (pre-May 2026, retained from previous archival
passes)

- `check_import.py`, `check_parquet_fens.py`, `compute_swa.py`,
  `create_openings.py`, `debug*.py`, `debug_init_bias.py`,
  `debug_network_bias.py`, `distill_buffer.py`, `init_small_model.py`,
  `inspect_null_policies.py`, `mate_in_one_fens.py`,
  `one_mcts_inference.py`, `read_buffer.py`, `test_small_model.py`.

## Why archive instead of delete?

These scripts were in `git` history; deleting them would lose that
history. Moving them to `archive/` keeps the git log intact while
removing them from the active `scripts/` namespace so nothing new
gets built on top of an investigation that has run its course.
