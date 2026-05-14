#pragma once

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "chess-library/include/chess.hpp"

#include "inference.hpp"
#include "mcts/common.hpp"

namespace chessrl::mcts::single
{

template <typename Result>
class BatchTicket
{
public:
    BatchTicket() = default;
    explicit BatchTicket(std::shared_future<Result> future)
        : future_(std::move(future))
    {
    }

    bool valid() const noexcept
    {
        return future_.valid();
    }

    bool ready() const
    {
        if (!future_.valid())
        {
            return false;
        }
        return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    Result get() const
    {
        return future_.get();
    }

private:
    std::shared_future<Result> future_;
};

struct SmallEncodedPosition
{
    std::vector<float> features;
};

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
    std::string engine_path{"model.trt"};
    int max_batch_size{1024};
    bool enable_telemetry{false};
};

class SmallBatchRunner
{
public:
    using Ticket = BatchTicket<SmallPolicyValue>;

    explicit SmallBatchRunner(SmallBatchConfig config = {});

    Ticket enqueue(SmallEncodedPosition position);
    void process_pending();
    size_t pending() const;
    BatchStatistics statistics() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace chessrl::mcts::single


