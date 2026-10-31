#include "mcts/single_inference.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mcts/small_layout.hpp"
#include "net/encoder_net.hpp"

namespace chessrl::mcts::single
{
namespace
{

std::string resolve_path(const std::string& path)
{
    try
    {
        return std::filesystem::absolute(path).string();
    }
    catch (...)
    {
        return path;
    }
}

class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
    {
        if (cudaMalloc(&mPtr, bytes) != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed.");
        }
    }

    ~DeviceBuffer()
    {
        if (mPtr)
        {
            cudaFree(mPtr);
        }
    }

    void* data() const noexcept { return mPtr; }

private:
    void* mPtr{nullptr};
};

class ResizableDeviceBuffer
{
public:
    void ensure(size_t bytes)
    {
        if (bytes == 0)
        {
            buffer_.reset();
            capacity_ = 0;
            return;
        }
        if (bytes > capacity_)
        {
            buffer_ = std::make_unique<DeviceBuffer>(bytes);
            capacity_ = bytes;
        }
    }

    void* data() const noexcept
    {
        return buffer_ ? buffer_->data() : nullptr;
    }

    size_t capacity() const noexcept { return capacity_; }

private:
    size_t capacity_{0};
    std::unique_ptr<DeviceBuffer> buffer_;
};

class PinnedBuffer
{
public:
    PinnedBuffer() = default;

    ~PinnedBuffer()
    {
        if (mPtr)
        {
            cudaFreeHost(mPtr);
        }
    }

    void ensure(size_t bytes)
    {
        if (bytes == 0)
        {
            if (mPtr)
            {
                cudaFreeHost(mPtr);
                mPtr = nullptr;
                capacity_ = 0;
            }
            return;
        }
        if (bytes > capacity_)
        {
            if (mPtr)
            {
                cudaFreeHost(mPtr);
            }
            if (cudaHostAlloc(&mPtr, bytes, cudaHostAllocDefault) != cudaSuccess)
            {
                throw std::runtime_error("cudaHostAlloc failed.");
            }
            capacity_ = bytes;
        }
    }

    void* data() const noexcept { return mPtr; }

    size_t capacity() const noexcept { return capacity_; }

    // Non-copyable, movable
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : mPtr(other.mPtr)
        , capacity_(other.capacity_)
    {
        other.mPtr = nullptr;
        other.capacity_ = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept
    {
        if (this != &other)
        {
            if (mPtr)
            {
                cudaFreeHost(mPtr);
            }
            mPtr = other.mPtr;
            capacity_ = other.capacity_;
            other.mPtr = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }

private:
    void* mPtr{nullptr};
    size_t capacity_{0};
};

class CudaStream
{
public:
    CudaStream()
    {
        // Non-blocking: required for stream capture (a stream that implicitly
        // syncs with the legacy default stream cannot be captured). Also
        // removes false serialization against the default stream.
        if (cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking) != cudaSuccess)
        {
            throw std::runtime_error("Failed to create CUDA stream.");
        }
    }

    ~CudaStream()
    {
        if (mStream)
        {
            cudaStreamDestroy(mStream);
        }
    }

    cudaStream_t get() const noexcept { return mStream; }

private:
    cudaStream_t mStream{};
};

template <typename Pending>
void fulfillWithException(std::vector<Pending>& work, const std::exception_ptr& ep)
{
    for (auto& item : work)
    {
        if (item.slot && !item.slot->delivered)
        {
            item.slot->error = ep;
            item.slot->delivered = true;
            item.slot->group->done();
        }
    }
}

} // namespace

struct SmallBatchRunner::Impl
{
    struct SmallPending
    {
        // POINTER, not an inline 776 B POD. The shared queue is guarded by a
        // single mutex hit by 128+ MCTS producers; an inline 776 B payload
        // bloated that critical section ~20× and serialized the producers
        // (the measured 2.2× regression). A pointer keeps enqueue/drain
        // pointer-sized, like the original dense vector-handle path.
        std::shared_ptr<const SmallPackedPosition> position;
        std::shared_ptr<EvalSlot> slot;
    };

    // A pinned buffer the GPU copies a batch's records into; the search threads read their
    // records from it directly. `readers` counts the records not yet taken: the worker reuses
    // the buffer only once it is back to 0. It never waits for one: a search holds the records
    // already back while it waits for the rest of its round, so waiting could deadlock. When no
    // buffer is free, the batch goes through `spill` and its results are handed back by value.
    struct OutputBuffer
    {
        PinnedBuffer host;
        std::atomic<int> readers{0};
    };
    static constexpr int kOutputBuffers = 16;  // per worker

    // Per-worker state: a CUDA stream and the staging buffers. Workers pop sub-batches
    // from the shared queue independently, so while one batch runs on the GPU the next
    // worker is already packing its own.
    struct Worker
    {
        std::thread thread;
        std::unique_ptr<CudaStream> stream;
        ResizableDeviceBuffer packedDevice;  // ~776 B per position
        PinnedBuffer hostPacked;
        ResizableDeviceBuffer outDevice;     // one kRecordBytes record per position
        std::array<OutputBuffer, kOutputBuffers> outputs;
        int next_output{0};
        PinnedBuffer spill;                  // a batch's records when no output buffer is free
        cudaEvent_t done{nullptr};           // blocking-sync event (CHESSRL_BLOCKING_SYNC)
    };

    explicit Impl(SmallBatchConfig cfg)
        : config(std::move(cfg))
        , net(config.engine_path)
        , flush_threshold_(static_cast<size_t>(config.max_batch_size))
    {
        if (config.max_batch_size <= 0)
        {
            throw std::invalid_argument("SmallBatchRunner requires positive max_batch_size");
        }

        // Number of concurrent inference workers (CHESSRL_INFER_WORKERS, default 4).
        int n = 4;
        if (const char* env = std::getenv("CHESSRL_INFER_WORKERS"))
        {
            const int parsed = std::atoi(env);
            if (parsed > 0) n = parsed;
        }
        workers_.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            workers_.emplace_back(std::make_unique<Worker>());
        }
        for (auto& w : workers_)
        {
            Worker* raw = w.get();
            raw->thread = std::thread([this, raw]() { worker_loop(*raw); });
        }
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop_worker_ = true;
        }
        cv.notify_all();
        for (auto& w : workers_)
        {
            if (w && w->thread.joinable())
            {
                w->thread.join();
            }
            if (w && w->done)
            {
                cudaEventDestroy(w->done);
            }
        }
        // Results still referring to the output buffers must be taken (or dropped) first; the
        // searches do so before the runner goes away. If one never is, leak the buffers rather
        // than free memory a result still points to.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (auto& w : workers_)
        {
            for (auto& out : w->outputs)
            {
                while (out.readers.load(std::memory_order_acquire) > 0)
                {
                    if (std::chrono::steady_clock::now() > deadline)
                    {
                        std::cerr << "[runner] results still pending at shutdown; keeping their buffers\n";
                        w.release();
                        break;
                    }
                    std::this_thread::yield();
                }
                if (!w) break;
            }
        }
        if (telemetry_enabled_ && batches_ > 0)
        {
            const double b = static_cast<double>(batches_);
            std::cerr << "[runner] " << batches_ << " batches, " << static_cast<double>(positions_) / b
                      << " positions each; per batch: GPU and copies " << total_infer_ms_ / b
                      << " ms, results handed back " << total_handback_ms_ / b << " ms, waiting for work "
                      << total_wait_ms_ / b << " ms (" << workers_.size() << " workers, " << spilled_
                      << " batches handed back by value)\n";
        }
    }

    EvalTicket enqueue(std::shared_ptr<const SmallPackedPosition> position, EvalGroup& group)
    {
        auto slot = std::make_shared<EvalSlot>();
        slot->group = &group;
        group.add();
        EvalTicket ticket(slot);
        size_t before = 0;
        size_t after = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            before = queue.size();
            queue.emplace_back(SmallPending{std::move(position), std::move(slot)});
            after = queue.size();
        }
        wake_for(before, after);
        return ticket;
    }

    void enqueue(std::vector<std::shared_ptr<const SmallPackedPosition>>& positions, EvalGroup& group,
                 std::vector<EvalTicket>& tickets)
    {
        if (positions.empty()) return;
        thread_local std::vector<std::shared_ptr<EvalSlot>> slots;
        slots.clear();
        group.add(static_cast<int>(positions.size()));
        for (size_t i = 0; i < positions.size(); ++i)
        {
            auto slot = std::make_shared<EvalSlot>();
            slot->group = &group;
            tickets.emplace_back(slot);
            slots.push_back(std::move(slot));
        }
        size_t before = 0;
        size_t after = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            before = queue.size();
            for (size_t i = 0; i < positions.size(); ++i)
            {
                queue.emplace_back(SmallPending{std::move(positions[i]), std::move(slots[i])});
            }
            after = queue.size();
        }
        positions.clear();
        wake_for(before, after);
    }

    // Wake a worker only when there is something new for it: the queue was empty (workers
    // sleep until there is work) or it just reached a full batch (a worker may be waiting for
    // one). Waking one on every position only had it check the queue and sleep again. A worker
    // that leaves positions in the queue wakes the next one (process()).
    void wake_for(size_t before, size_t after)
    {
        if (before == 0 || (before < flush_threshold_ && after >= flush_threshold_)) cv.notify_one();
    }

    void process(Worker& w)
    {
        std::vector<SmallPending> work;
        work.reserve(static_cast<size_t>(config.max_batch_size));
        bool left = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            while (!queue.empty() && work.size() < static_cast<size_t>(config.max_batch_size))
            {
                work.push_back(std::move(queue.front()));
                queue.pop_front();
            }
            left = !queue.empty();
        }
        if (left) cv.notify_one();
        if (work.empty())
        {
            return;
        }
        try
        {
            run_batch(w, work, std::chrono::steady_clock::now());
        }
        catch (...)
        {
            fulfillWithException(work, std::current_exception());
            throw;
        }
    }

    // The packed positions in, one kernel, one record per position out (FP16 policy
    // logits, FP32 WDL, mlh, mlh_q). Batches run at their exact size.
    void run_batch(Worker& w, std::vector<SmallPending>& work,
                   std::chrono::steady_clock::time_point start_time)
    {
        using chessrl::net::kAuxOffset;
        using chessrl::net::kRecordBytes;
        if (!w.stream) w.stream = std::make_unique<CudaStream>();
        const size_t n = work.size();
        const size_t cap = static_cast<size_t>(config.max_batch_size);
        w.packedDevice.ensure(cap * sizeof(SmallPackedPosition));
        w.hostPacked.ensure(cap * sizeof(SmallPackedPosition));
        w.outDevice.ensure(cap * kRecordBytes);
        OutputBuffer* out = nullptr;
        for (int k = 0; k < kOutputBuffers && !out; ++k)
        {
            OutputBuffer& candidate = w.outputs[static_cast<size_t>((w.next_output + k) % kOutputBuffers)];
            if (candidate.readers.load(std::memory_order_acquire) == 0)
            {
                out = &candidate;
                w.next_output = (w.next_output + k + 1) % kOutputBuffers;
            }
        }
        PinnedBuffer& dst = out ? out->host : w.spill;
        dst.ensure(cap * kRecordBytes);
        auto* hp = static_cast<SmallPackedPosition*>(w.hostPacked.data());
        for (size_t i = 0; i < n; ++i) hp[i] = *work[i].position;

        const cudaStream_t s = w.stream->get();
        cudaError_t err = cudaMemcpyAsync(w.packedDevice.data(), hp, n * sizeof(SmallPackedPosition),
                                          cudaMemcpyHostToDevice, s);
        if (err == cudaSuccess)
        {
            err = net.forward(static_cast<const SmallPackedPosition*>(w.packedDevice.data()), nullptr,
                              static_cast<int>(n), w.outDevice.data(), s);
        }
        if (err == cudaSuccess)
        {
            err = cudaMemcpyAsync(dst.data(), w.outDevice.data(), n * kRecordBytes,
                                  cudaMemcpyDeviceToHost, s);
        }
        if (err == cudaSuccess)
        {
            // Spinning in cudaStreamSynchronize keeps a core busy while the GPU works;
            // a blocking-sync event puts the thread to sleep instead.
            if (blocking_sync_)
            {
                if (!w.done)
                {
                    err = cudaEventCreateWithFlags(&w.done, cudaEventBlockingSync | cudaEventDisableTiming);
                }
                if (err == cudaSuccess) err = cudaEventRecord(w.done, s);
                if (err == cudaSuccess) err = cudaEventSynchronize(w.done);
            }
            else
            {
                err = cudaStreamSynchronize(s);
            }
        }
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("fused network batch failed: ") + cudaGetErrorString(err));
        }
        const auto synced = std::chrono::steady_clock::now();
        if (telemetry_enabled_)
        {
            const double infer_ms = std::chrono::duration<double, std::milli>(synced - start_time).count();
            std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
            batches_ += 1;
            positions_ += n;
            total_infer_ms_ += infer_ms;
            if (!out) spilled_ += 1;
        }
        const auto* rec = static_cast<const unsigned char*>(dst.data());
        if (out) out->readers.store(static_cast<int>(n), std::memory_order_relaxed);
        for (size_t i = 0; i < n; ++i, rec += kRecordBytes)
        {
            EvalSlot& slot = *work[i].slot;
            const auto* policy = reinterpret_cast<const std::uint16_t*>(rec);
            const auto* aux = reinterpret_cast<const float*>(rec + kAuxOffset);
            if (out)
            {
                slot.policy_fp16 = policy;
                slot.aux = aux;
                slot.policy_size = kSmallPolicySize;
                slot.readers = &out->readers;
            }
            else
            {
                slot.value.policy_logits.resize(kSmallPolicySize);
                fp16_to_float(policy, slot.value.policy_logits.data(), kSmallPolicySize);
                slot.value.wdl = {aux[0], aux[1], aux[2]};
                slot.value.mlh = aux[3];
                slot.value.mlh_q = aux[4];
            }
            slot.delivered = true;
            slot.group->done();  // wakes the search only when this was its last position
        }
        if (telemetry_enabled_)
        {
            const double handback_ms = std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - synced)
                                           .count();
            std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
            total_handback_ms_ += handback_ms;
        }
    }

    size_t pending_size() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }

    void notify_all()
    {
        cv.notify_all();
    }

    void worker_loop(Worker& w)
    {
        while (true)
        {
            const auto idle_from = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(mutex);

            // Stage 1: Idle wait - sleep until queue has work or we're stopping.
            cv.wait(lock, [this]() { return stop_worker_ || !queue.empty(); });

            if (stop_worker_ && queue.empty())
            {
                break;
            }

            // Stage 2: Batching wait - if the queue is below the flush
            // threshold, give it a brief window to fill so we get larger,
            // GPU-efficient batches. With multiple workers, peers may also be
            // waiting on this same queue; once one of them takes a batch the
            // rest re-evaluate. Workers that are actively in process() (lock
            // released) are not counted here, which is fine — having one
            // worker take a small batch while another waits a bit longer for
            // a bigger one is exactly the desired pipelining behavior.
            if (queue.size() < flush_threshold_)
            {
                cv.wait_for(lock, flush_timeout_, [this]() {
                    return stop_worker_ || queue.size() >= flush_threshold_;
                });
            }

            // Stage 3: Process - unlock before calling process() since it locks internally.
            lock.unlock();
            if (telemetry_enabled_)
            {
                const double wait_ms = std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - idle_from)
                                           .count();
                std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
                total_wait_ms_ += wait_ms;
            }

            try
            {
                process(w);
            }
            catch (...)
            {
                // Exceptions are delivered through the result slots; keep the worker alive.
            }
        }
    }

    BatchStatistics stats() const
    {
        std::lock_guard<std::mutex> lock(telemetry_mutex_);
        return BatchStatistics{batches_, positions_, total_infer_ms_};
    }

    SmallBatchConfig config;
    const chessrl::net::EncoderNet net;
    size_t flush_threshold_{0};
    // Stable storage via unique_ptr so worker threads can hold raw references.
    std::vector<std::unique_ptr<Worker>> workers_;
    mutable std::mutex mutex;
    std::condition_variable cv;
    bool stop_worker_{false};
    std::deque<SmallPending> queue;
    // How long a worker waits for a partly filled queue to reach max_batch_size
    // (CHESSRL_MCTS_FLUSH_US, default 100 µs).
    const std::chrono::microseconds flush_timeout_{[](){
        if (const char* env = std::getenv("CHESSRL_MCTS_FLUSH_US")) return std::atoi(env);
        return 100;
    }()};
    // Wait for each batch on a blocking-sync event rather than spinning
    // (CHESSRL_BLOCKING_SYNC=1).
    const bool blocking_sync_{[](){
        const char* env = std::getenv("CHESSRL_BLOCKING_SYNC");
        return env != nullptr && std::atoi(env) != 0;
    }()};
    const bool telemetry_enabled_{config.enable_telemetry};
    mutable std::mutex telemetry_mutex_;
    size_t batches_{0};
    size_t positions_{0};
    double total_infer_ms_{0.0};
    double total_handback_ms_{0.0};  // from the batch's sync to its last result handed back
    double total_wait_ms_{0.0};      // a worker waiting for a batch (idle and flush windows)
    size_t spilled_{0};              // batches that found no free output buffer
};

SmallBatchRunner::SmallBatchRunner(SmallBatchConfig config)
{
    struct Cache
    {
        std::mutex mutex;
        std::unordered_map<std::string, std::weak_ptr<Impl>> entries;
    };
    static Cache cache;

    const std::string key = [&config]() {
        std::ostringstream oss;
        oss << resolve_path(config.engine_path)
            << '|' << config.max_batch_size
            << '|' << (config.enable_telemetry ? '1' : '0');
        return oss.str();
    }();

    std::lock_guard<std::mutex> lock(cache.mutex);
    if (auto existing = cache.entries[key].lock())
    {
        impl_ = existing;
        return;
    }

    auto fresh = std::make_shared<Impl>(std::move(config));
    cache.entries[key] = fresh;
    impl_ = std::move(fresh);
}

SmallBatchRunner::Ticket SmallBatchRunner::enqueue(
    std::shared_ptr<const SmallPackedPosition> position, EvalGroup& group)
{
    return impl_->enqueue(std::move(position), group);
}

void SmallBatchRunner::enqueue(std::vector<std::shared_ptr<const SmallPackedPosition>>& positions,
                               EvalGroup& group, std::vector<Ticket>& tickets)
{
    impl_->enqueue(positions, group, tickets);
}

void SmallBatchRunner::process_pending()
{
    // Workers are always running; nudge any that may be sleeping to drain
    // whatever's in the queue. Kept as a no-arg signal for API stability;
    // inference itself is dispatched by the worker threads on their own
    // streams/contexts.
    impl_->notify_all();
}

size_t SmallBatchRunner::pending() const
{
    return impl_->pending_size();
}

BatchStatistics SmallBatchRunner::statistics() const
{
    return impl_->stats();
}

} // namespace chessrl::mcts::single

