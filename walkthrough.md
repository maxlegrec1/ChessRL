# Buffer Size Discrepancy Fix Walkthrough

## Changes
I identified that the C++ code was using an incorrect record size constant (100 bytes) to estimate the buffer size, while the actual record size is 128 bytes. This caused the TUI to report ~28% more positions than actually existed.

I modified `cpp/selfplay.cpp` to use the correct record size of 128 bytes.

## Verification
The `read_buffer.py` script correctly uses 128 bytes per record.
With the fix, the C++ `get_buffer_size` function now also uses 128 bytes.

```cpp
// cpp/selfplay.cpp
size_t SelfPlayWorker::get_buffer_size() const
{
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
