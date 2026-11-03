#include "bench/fake_network.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <immintrin.h>

#include "inference.hpp"
#include "mcts/single_inference.hpp"

namespace chessrl::bench
{
namespace detail
{

constexpr std::size_t kTables = 1024; // distinct logit vectors; a position picks one by hash

struct FakeNetwork
{
    FakeNetworkConfig config;
    std::size_t policy_size{0};
    std::vector<float> logits; // kTables x policy_size
    // The delayed (self-play) path's results, as the real runner leaves them in its output buffer: FP16 logits and
    // the 5 aux floats (WDL, mlh, mlh_q) per table row.
    std::vector<std::uint16_t> logits_fp16;
    std::vector<float> aux;
    std::atomic<std::uint64_t> evaluations{0};
    std::atomic<std::uint64_t> batches{0};
};

FakeNetwork& network()
{
    static FakeNetwork net;
    return net;
}

std::uint64_t mix(std::uint64_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Hash of the current position: slot-0 pieces, side to move, castling, en passant.
std::uint64_t position_hash(const mcts::single::SmallPackedPosition& p)
{
    std::uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int k = 0; k < 12; ++k)
    {
        h = mix(h ^ p.bitboards[0][k]) + static_cast<std::uint64_t>(k);
    }
    const std::uint64_t flags = static_cast<std::uint64_t>(p.stm)
        | (static_cast<std::uint64_t>(p.castling) << 8)
        | (static_cast<std::uint64_t>(static_cast<std::uint16_t>(p.ep_square)) << 16);
    return mix(h ^ flags);
}

float unit(std::uint64_t h) // [0, 1)
{
    return static_cast<float>(static_cast<double>(h >> 40) / 16777216.0);
}

SmallPolicyValue evaluate(const mcts::single::SmallPackedPosition& p)
{
    FakeNetwork& net = network();
    const std::uint64_t h = position_hash(p);
    SmallPolicyValue out;
    const float* row = net.logits.data() + (h % kTables) * net.policy_size;
    out.policy_logits.assign(row, row + net.policy_size);
    const float lean = 2.0F * unit(mix(h + 1)) - 1.0F; // White-Black balance of the decisive part
    const float draw = 0.2F + 0.5F * unit(mix(h + 2));
    out.wdl = {(1.0F - draw) * (1.0F + lean) / 2.0F, draw, (1.0F - draw) * (1.0F - lean) / 2.0F};
    out.mlh = out.mlh_q = 0.4F + 2.4F * unit(mix(h + 3)); // 10 to 70 plies, in units of 25
    net.evaluations.fetch_add(1, std::memory_order_relaxed);
    return out;
}

} // namespace detail

void configure_fake_network(const FakeNetworkConfig& config)
{
    detail::FakeNetwork& net = detail::network();
    net.config = config;
    net.policy_size = small_policy_size();
    net.logits.resize(detail::kTables * net.policy_size);
    std::mt19937_64 rng(config.seed);
    std::normal_distribution<float> normal(0.0F, config.sharpness);
    for (float& x : net.logits)
    {
        x = normal(rng);
    }
    net.logits_fp16.resize(net.logits.size());
    for (std::size_t i = 0; i < net.logits.size(); ++i)
    {
        net.logits_fp16[i] = _cvtss_sh(net.logits[i], 0);
    }
    net.aux.resize(detail::kTables * 5);
    for (std::size_t r = 0; r < detail::kTables; ++r)
    {
        const std::uint64_t h = detail::mix(r + 0x51ED);
        const float lean = 2.0F * detail::unit(detail::mix(h + 1)) - 1.0F;
        const float draw = 0.2F + 0.5F * detail::unit(detail::mix(h + 2));
        float* a = net.aux.data() + 5 * r;
        a[0] = (1.0F - draw) * (1.0F + lean) / 2.0F;
        a[1] = draw;
        a[2] = (1.0F - draw) * (1.0F - lean) / 2.0F;
        a[3] = a[4] = 0.4F + 2.4F * detail::unit(detail::mix(h + 3));
    }
}

FakeNetworkStats fake_network_stats()
{
    const detail::FakeNetwork& net = detail::network();
    return {net.evaluations.load(), net.batches.load()};
}

double fake_network_cost_ns(const std::vector<mcts::single::SmallPackedPosition>& positions, int reps)
{
    detail::FakeNetwork& net = detail::network();
    const std::uint64_t before = net.evaluations.load();
    float sink = 0.0F;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
    {
        for (const auto& p : positions)
        {
            sink += detail::evaluate(p).wdl[0];
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    net.evaluations.store(before); // not search evaluations
    const double n = static_cast<double>(reps) * static_cast<double>(positions.size());
    volatile float keep = sink;
    (void)keep;
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / n;
}

} // namespace chessrl::bench

namespace chessrl::mcts::single
{

struct SmallBatchRunner::Impl
{
    struct Request
    {
        std::shared_ptr<const SmallPackedPosition> position;
        std::shared_ptr<EvalSlot> slot;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Request> queue;
    bool stop{false};
    BatchStatistics stats;
    // As the real runner: CHESSRL_INFER_WORKERS dispatchers (default 6), each waiting out its batch's simulated
    // GPU time spinning (cudaStreamSynchronize) or, with CHESSRL_BLOCKING_SYNC=1, asleep.
    std::vector<std::thread> dispatchers;
    const bool blocking_sync{[] {
        const char* env = std::getenv("CHESSRL_BLOCKING_SYNC");
        return env != nullptr && std::atoi(env) != 0;
    }()};

    static int worker_count()
    {
        const char* env = std::getenv("CHESSRL_INFER_WORKERS");
        const int n = env ? std::atoi(env) : 6;
        return n > 0 ? n : 6;
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        for (auto& d : dispatchers)
        {
            if (d.joinable()) d.join();
        }
    }

    // Simulated GPU: batch like the real runner, then answer after delay_us.
    void dispatch_loop()
    {
        const bench::FakeNetworkConfig& cfg = bench::detail::network().config;
        std::unique_lock<std::mutex> lock(mutex);
        while (true)
        {
            cv.wait(lock, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty())
            {
                return;
            }
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(cfg.flush_us);
            cv.wait_until(lock, deadline, [&] {
                return stop || queue.size() >= static_cast<std::size_t>(cfg.max_batch);
            });
            std::vector<Request> batch;
            while (!queue.empty() && batch.size() < static_cast<std::size_t>(cfg.max_batch))
            {
                batch.push_back(std::move(queue.front()));
                queue.pop_front();
            }
            const bool more = !queue.empty();
            lock.unlock();
            if (more) cv.notify_one();  // the next dispatcher takes the rest, as the real workers do
            const auto ready = std::chrono::steady_clock::now() + std::chrono::microseconds(cfg.delay_us);
            if (blocking_sync)
            {
                std::this_thread::sleep_until(ready);
            }
            else
            {
                while (std::chrono::steady_clock::now() < ready)
                {
                }
            }
            bench::detail::FakeNetwork& net = bench::detail::network();
            for (auto& r : batch)
            {
                const std::size_t row = bench::detail::position_hash(*r.position) % bench::detail::kTables;
                r.slot->policy_fp16 = net.logits_fp16.data() + row * net.policy_size;
                r.slot->policy_size = static_cast<int>(net.policy_size);
                r.slot->aux = net.aux.data() + 5 * row;
                r.slot->delivered = true;
                r.slot->group->done();
            }
            net.evaluations.fetch_add(batch.size(), std::memory_order_relaxed);
            bench::detail::network().batches.fetch_add(1, std::memory_order_relaxed);
            lock.lock();
            stats.batches += 1;
            stats.positions += batch.size();
            stats.total_inference_ms += cfg.delay_us / 1000.0;
        }
    }
};

SmallBatchRunner::SmallBatchRunner(SmallBatchConfig /*config*/)
    : impl_(std::make_shared<Impl>())
{
    if (bench::detail::network().config.delay_us > 0)
    {
        for (int i = 0; i < Impl::worker_count(); ++i)
        {
            impl_->dispatchers.emplace_back([impl = impl_.get()] { impl->dispatch_loop(); });
        }
    }
}

SmallBatchRunner::Ticket SmallBatchRunner::enqueue(std::shared_ptr<const SmallPackedPosition> position,
                                                  EvalGroup& group)
{
    auto slot = std::make_shared<EvalSlot>();
    slot->group = &group;
    group.add();
    Ticket ticket(slot);
    if (bench::detail::network().config.delay_us <= 0)
    {
        slot->value = bench::detail::evaluate(*position);
        slot->delivered = true;
        group.done();
        return ticket;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->queue.push_back({std::move(position), std::move(slot)});
    }
    impl_->cv.notify_one();
    return ticket;
}

void SmallBatchRunner::enqueue(std::vector<std::shared_ptr<const SmallPackedPosition>>& positions,
                               EvalGroup& group, std::vector<Ticket>& tickets)
{
    if (bench::detail::network().config.delay_us <= 0)
    {
        for (auto& position : positions)
        {
            tickets.push_back(enqueue(std::move(position), group));
        }
        positions.clear();
        return;
    }
    // As the real runner: one lock for the round, a wake-up only when the queue was empty or reaches a batch.
    if (positions.empty()) return;
    group.add(static_cast<int>(positions.size()));
    std::vector<std::shared_ptr<EvalSlot>> slots;
    slots.reserve(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        auto slot = std::make_shared<EvalSlot>();
        slot->group = &group;
        tickets.emplace_back(slot);
        slots.push_back(std::move(slot));
    }
    std::size_t before = 0;
    std::size_t after = 0;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        before = impl_->queue.size();
        for (std::size_t i = 0; i < positions.size(); ++i)
        {
            impl_->queue.push_back({std::move(positions[i]), std::move(slots[i])});
        }
        after = impl_->queue.size();
    }
    positions.clear();
    const auto max_batch = static_cast<std::size_t>(bench::detail::network().config.max_batch);
    if (before == 0 || (before < max_batch && after >= max_batch)) impl_->cv.notify_one();
}

void SmallBatchRunner::process_pending() {}

size_t SmallBatchRunner::pending() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->queue.size();
}

BatchStatistics SmallBatchRunner::statistics() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats;
}

} // namespace chessrl::mcts::single
