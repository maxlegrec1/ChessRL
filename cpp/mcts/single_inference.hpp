#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#if defined(__F16C__)
#include <immintrin.h>
#endif

#include "chess-library/include/chess.hpp"

#include "inference.hpp"
#include "mcts/common.hpp"
// SmallPackedPosition + plane-layout constants: standalone header (only
// <cstdint>) shared with the fused CUDA kernel so nvcc never parses
// chess-library.
#include "mcts/packed_position.hpp"

namespace chessrl::mcts::single
{

// The evaluations one search has in flight. The runner wakes the search once, when the last of
// them is ready, instead of once per position (a wake-up per position cost the inference
// workers 1.5 µs each). Every change to the count is made under the lock, so once wait() has
// returned no worker touches the group again.
class EvalGroup
{
public:
    EvalGroup() = default;
    EvalGroup(const EvalGroup&) = delete;
    EvalGroup& operator=(const EvalGroup&) = delete;
    ~EvalGroup() { wait(); }

    void add(int n = 1)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ += n;
    }

    void done()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--pending_ == 0) cv_.notify_one();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return pending_ == 0; });
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int pending_{0};
};

// One network evaluation in flight. The runner points it at the position's record in its pinned
// output buffer (FP16 policy logits, then WDL, mlh and mlh_q as floats) and the search thread
// converts that record itself when it takes the result, so the worker writes nothing per position;
// the buffer's `readers` counts the records not yet taken. The CPU benchmark's fake network fills
// `value` instead.
struct EvalSlot
{
    EvalGroup* group{nullptr};
    const std::uint16_t* policy_fp16{nullptr};
    const float* aux{nullptr};
    int policy_size{0};
    std::atomic<int>* readers{nullptr};
    bool delivered{false};
    bool taken{false};
    SmallPolicyValue value;
    std::exception_ptr error;

    ~EvalSlot()
    {
        if (readers && !taken) readers->fetch_sub(1, std::memory_order_release);
    }
};

inline void fp16_to_float(const std::uint16_t* src, float* dst, size_t n)
{
    size_t i = 0;
#if defined(__F16C__)
    for (; i + 8 <= n; i += 8)
    {
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
    }
    for (; i < n; ++i) dst[i] = _cvtsh_ss(src[i]);
#else
    for (; i < n; ++i)
    {
        const std::uint32_t h = src[i], sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
        float f;
        if (exp == 0)
        {
            f = static_cast<float>(mant) * 5.9604645e-8f;  // subnormal: mant * 2^-24
            if (sign) f = -f;
        }
        else
        {
            const std::uint32_t bits = sign | ((exp == 31 ? 255u : exp + 112u) << 23) | (mant << 13);
            std::memcpy(&f, &bits, sizeof(f));
        }
        dst[i] = f;
    }
#endif
}

// A pending evaluation of `group`. Call get() once, after the group's wait() has returned.
class EvalTicket
{
public:
    EvalTicket() = default;
    explicit EvalTicket(std::shared_ptr<EvalSlot> slot)
        : slot_(std::move(slot))
    {
    }

    bool valid() const noexcept
    {
        return slot_ != nullptr;
    }

    SmallPolicyValue get() const
    {
        EvalSlot& s = *slot_;
        if (s.error) std::rethrow_exception(s.error);
        if (!s.policy_fp16) return std::move(s.value);
        SmallPolicyValue v;
        v.policy_logits.resize(static_cast<size_t>(s.policy_size));
        fp16_to_float(s.policy_fp16, v.policy_logits.data(), v.policy_logits.size());
        v.wdl = {s.aux[0], s.aux[1], s.aux[2]};
        v.mlh = s.aux[3];
        v.mlh_q = s.aux[4];
        s.taken = true;
        s.readers->fetch_sub(1, std::memory_order_release);
        return v;
    }

private:
    std::shared_ptr<EvalSlot> slot_;
};

struct SmallEncodedPosition
{
    std::vector<float> features;
};

// Pack a (board, history) pair into the compact snapshot.
SmallPackedPosition pack_small_position(
    const chess::Board& board, const chessrl::mcts::HistoryStack& history);

// CPU reference expansion — bit-identical to encode_small_position().features
// for the same logical position. The GPU kernel must reproduce this exactly.
std::vector<float> expand_packed_to_dense(const SmallPackedPosition& packed);

struct BatchStatistics
{
    size_t batches{0};
    size_t positions{0};
    double total_inference_ms{0.0};
};

// Process-wide encoder time accumulators. Updated by encodeSmallFeatures /
// encodeSmallFeaturesNoHistory; read by the perf bench. Atomic because the
// encoder runs on N MCTS worker threads concurrently.
struct EncoderTotals
{
    uint64_t calls{0};
    uint64_t nanos{0};
};
EncoderTotals encoder_totals();
void encoder_totals_reset();

// No-history convenience: equivalent to encode_small_position(board, {}).
SmallEncodedPosition encode_small_position(const chess::Board& board);

// History-aware encoder. The 0th slot of `history` is conceptually the
// current position; the encoder reads piece bitboards from `board` itself
// (defensive — keeps slot-0 piece planes authoritative even if the
// caller's ring buffer drifts) but uses `history[0].repetition` for the
// current-position repetition flag and `history[1..7]` for past slots.
SmallEncodedPosition encode_small_position(
    const chess::Board& board, const chessrl::mcts::HistoryStack& history);

struct SmallBatchConfig
{
    std::string engine_path{"model.cnet"};  // a .cnet file (src/export/to_cnet.py)
    int max_batch_size{1024};
    bool enable_telemetry{false};
};

class SmallBatchRunner
{
public:
    using Ticket = EvalTicket;

    explicit SmallBatchRunner(SmallBatchConfig config = {});

    // Carries a POINTER to the compact packed snapshot (keeps the shared-
    // queue critical section pointer-sized). The fused kernel expands it to
    // the input planes on the GPU. `group` is woken when all its positions are done.
    Ticket enqueue(std::shared_ptr<const SmallPackedPosition> position, EvalGroup& group);
    // A search round's positions at once (moved from `positions`): one lock on the shared queue
    // and at most one wake-up instead of one of each per position. Appends a ticket per position.
    void enqueue(std::vector<std::shared_ptr<const SmallPackedPosition>>& positions, EvalGroup& group,
                 std::vector<Ticket>& tickets);
    void process_pending();
    size_t pending() const;
    BatchStatistics statistics() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace chessrl::mcts::single


