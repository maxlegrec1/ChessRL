# Search performance log

CPU cost of the self-play search, measured with `scripts/bench_cpu_search.sh`: the production
search code (`cpp/mcts/mcts_small.cpp`) with a fake network in place of TensorRT
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

## Baseline (2026-10-29)

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
