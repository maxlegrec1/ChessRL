# Self-play perf: `phase0_long` vs `phase2c_best`

- GPU: NVIDIA H200 vs NVIDIA H200
- Engine A: `artifacts/legal_ceiling_v7_history_arch/v7.trt`  (50 steps in 304.3s)
- Engine B: `artifacts/legal_ceiling_v7_history_arch/v7.trt`  (84 steps in 241.3s)
- Config: concurrency=1024/1024  threads=256/128  sims=800/800

| metric | phase0_long | phase2c_best | Δ |
|---|---:|---:|---:|
| positions/sec | 168.27 | 356.40 | +111.8% |
| games/sec | 0.73 | 1.46 | +99.9% |
| nodes/sec | 134,618.58 | 285,117.31 | +111.8% |
| mean batch size | 46.54 | 127.32 | +173.6% |
| ms / batch | 0.88 | 0.99 | +12.0% |
| us / encoder leaf | 4.08 | 2.25 | -44.9% |
| encoder thread busy % | 0.21 | 0.49 | +134.8% |
