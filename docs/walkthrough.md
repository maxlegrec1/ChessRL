# Buffer Size Discrepancy Fix## Optimization Results

### Granular Profiling Analysis
Granular profiling revealed that **Inference Wait** was the dominant bottleneck (~2500ms out of ~2800ms total MCTS time), while GPU utilization remained low. This pointed to inefficient batching.

### Batching Optimization
We identified that the `SmallBatchRunner` was flushing batches too aggressively due to a combination of:
1.  Unreachable `flush_threshold` (default 1024 vs 128 threads).
2.  Extremely short `flush_timeout` (1us).

We optimized this by:
1.  Increasing `flush_timeout` to **500us** (configurable via `CHESSRL_MCTS_FLUSH_US`).
2.  Matching `max_batch_size` to the thread count (128) via `CHESSRL_MCTS_SMALL_BATCH_SIZE`.

### Final Performance
- **Throughput:** Increased from ~41 to **~53 positions/sec** (~30% improvement).
- **MCTS Time:** Reduced from ~3800ms to ~2500ms (with 500us timeout).
- **TRT Time:** Reduced significantly due to better batching efficiency.

The system is now CPU-bound by the MCTS simulation loop and thread synchronization, with GPU inference being efficiently batched.

## Verification
- **Self-Play:** Verified with `python -m src.self_play --boards 128 --threads 1`.
- **Telemetry:** Confirmed improved batch sizes and reduced inference wait times.
s 128 bytes per record.
With the fix, the C++ `get_buffer_size` function now also uses 128 bytes.

```cpp
// cpp/selfplay.cpp
size_t SelfPlayWorker::get_buffer_size() const
    // ...
    // Approximate: file_size / record_size
    // Record size is 128 bytes (100 FEN + 16 Policy + 12 Value)
    const size_t record_size = 128;
    size_t file_size = std::filesystem::file_size(p);
    return file_size / record_size;
}
```

The TUI should now report a buffer size that matches the output of `uv run read_buffer`.

## Statistics
Current buffer status from `read_buffer.py`:
- Total rows: 225,618
- Unique FENs: 221,878
- Win/Draw/Loss: ~29% / 41% / 29%
