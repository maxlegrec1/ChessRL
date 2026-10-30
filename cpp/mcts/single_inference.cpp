#include "mcts/single_inference.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
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

#include "cuda_fp16.h"

#if defined(__F16C__)
#include <immintrin.h>
#endif

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

void halfs_to_floats(const std::uint16_t* src, float* dst, size_t n)
{
    size_t i = 0;
#if defined(__F16C__)
    for (; i + 8 <= n; i += 8)
    {
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
    }
#endif
    for (; i < n; ++i)
    {
        __half h;
        std::memcpy(&h, src + i, sizeof(h));
        dst[i] = __half2float(h);
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
        if (item.promise)
        {
            item.promise->set_exception(ep);
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
        std::shared_ptr<std::promise<SmallPolicyValue>> promise;
    };

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
        PinnedBuffer hostOut;
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
        }
    }

    BatchTicket<SmallPolicyValue> enqueue(std::shared_ptr<const SmallPackedPosition> position)
    {
        auto promise = std::make_shared<std::promise<SmallPolicyValue>>();
        auto future = promise->get_future().share();
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.emplace_back(SmallPending{std::move(position), std::move(promise)});
        }
        cv.notify_one();
        return BatchTicket<SmallPolicyValue>(std::move(future));
    }

    void process(Worker& w)
    {
        std::vector<SmallPending> work;
        work.reserve(static_cast<size_t>(config.max_batch_size));
        {
            std::lock_guard<std::mutex> lock(mutex);
            while (!queue.empty() && work.size() < static_cast<size_t>(config.max_batch_size))
            {
                work.push_back(std::move(queue.front()));
                queue.pop_front();
            }
        }
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
        w.hostOut.ensure(cap * kRecordBytes);
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
            err = cudaMemcpyAsync(w.hostOut.data(), w.outDevice.data(), n * kRecordBytes,
                                  cudaMemcpyDeviceToHost, s);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(s);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("fused network batch failed: ") + cudaGetErrorString(err));
        }
        if (telemetry_enabled_)
        {
            const double infer_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - start_time)
                                        .count();
            std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
            batches_ += 1;
            positions_ += n;
            total_infer_ms_ += infer_ms;
        }
        const auto* rec = static_cast<const unsigned char*>(w.hostOut.data());
        for (size_t i = 0; i < n; ++i, rec += kRecordBytes)
        {
            SmallPolicyValue result;
            result.policy_logits.resize(kSmallPolicySize);
            halfs_to_floats(reinterpret_cast<const std::uint16_t*>(rec), result.policy_logits.data(),
                            kSmallPolicySize);
            const auto* aux = reinterpret_cast<const float*>(rec + kAuxOffset);
            result.wdl = {aux[0], aux[1], aux[2]};
            result.mlh = aux[3];
            result.mlh_q = aux[4];
            work[i].promise->set_value(std::move(result));
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

            try
            {
                process(w);
            }
            catch (...)
            {
                // Exceptions are delivered via promises; keep worker alive.
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
    const bool telemetry_enabled_{config.enable_telemetry};
    mutable std::mutex telemetry_mutex_;
    size_t batches_{0};
    size_t positions_{0};
    double total_infer_ms_{0.0};
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
    std::shared_ptr<const SmallPackedPosition> position)
{
    return impl_->enqueue(std::move(position));
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

