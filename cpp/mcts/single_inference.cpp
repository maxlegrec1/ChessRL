#include "mcts/single_inference.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_fp16.h"

#include "chess-library/include/chess.hpp"
#include "include/trt_utils.hpp"

namespace chessrl::mcts::single
{
namespace
{

using chessrl::trt::Logger;
using chessrl::trt::element_size;
using chessrl::trt::volume;
using chessrl::trt::resolve_engine_path;

constexpr int kSmallBoardSize = 8;
constexpr int kSmallFeaturePlanes = 19;
constexpr int kSmallPolicySize = 1929;
constexpr int kSmallValueSize = 3;

constexpr float kIllegalScore = -1e9F;

// Volume excluding the batch dimension (works for both static and dynamic shapes).
size_t normalized_tensor_volume(const nvinfer1::Dims& dims)
{
    if (dims.nbDims == 0) return 0;
    
    size_t result = 1;
    // Start from index 1 to skip the batch dimension (index 0)
    // This works for both Static [N, C, H, W] and Dynamic [-1, C, H, W]
    for (int i = 1; i < dims.nbDims; ++i)
    {
        result *= static_cast<size_t>(dims.d[i]);
    }
    return result;
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
        if (cudaStreamCreate(&mStream) != cudaSuccess)
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

// -------------------------------------------------------------
// Small-model single-batch inference
// -------------------------------------------------------------

struct SmallRawOutputs
{
    std::vector<float> policy;
    std::vector<float> wdl;
    std::vector<float> mlh;   // 1 element per batch row (true-target head)
    std::vector<float> mlh_q; // 1 element per batch row (MCTS-backed-target head)
};

class SmallEngineContext
{
public:
    explicit SmallEngineContext(std::string enginePath)
        : mEnginePath(std::move(enginePath))
    {
        std::ifstream engineFile(mEnginePath, std::ios::binary);
        if (!engineFile)
        {
            throw std::runtime_error(std::string("Failed to open engine file: ") + mEnginePath);
        }

        engineFile.seekg(0, std::ifstream::end);
        const auto fsize = engineFile.tellg();
        engineFile.seekg(0, std::ifstream::beg);

        std::vector<char> engineData(static_cast<size_t>(fsize));
        engineFile.read(engineData.data(), fsize);

        mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
        if (!mRuntime)
        {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!mEngine)
        {
            throw std::runtime_error("Failed to deserialize TensorRT engine.");
        }

        const int nbTensors = mEngine->getNbIOTensors();
        // First pass: pick out input + non-scalar outputs by shape, and
        // collect all scalar-per-row output names to disambiguate by name
        // afterwards (mlh vs mlh_q are both shape-1).
        std::vector<std::string> scalarOutputs;
        for (int i = 0; i < nbTensors; ++i)
        {
            const char* name = mEngine->getIOTensorName(i);
            const auto mode = mEngine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                mInputName = name;
            }
            else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
            {
                const auto dims = mEngine->getTensorShape(name);
                const size_t base = normalized_tensor_volume(dims);
                if (base == static_cast<size_t>(kSmallPolicySize) && mPolicyOutputName.empty())
                {
                    mPolicyOutputName = name;
                }
                else if (base == static_cast<size_t>(kSmallValueSize) && mValueOutputName.empty())
                {
                    mValueOutputName = name;
                }
                else if (base == 1)
                {
                    scalarOutputs.emplace_back(name);
                }
            }
        }

        // Disambiguate the two scalar heads by name. The exporter sets
        // output_names = ["policy", "value", "mlh", "mlh_q"], so we can match
        // exactly. If only one scalar exists (legacy single-head engine), it's
        // the true-target head; mlh_q stays empty and the runtime falls back
        // to using mlh as the leaf seed (degraded but not broken).
        for (const auto& name : scalarOutputs)
        {
            if (name == "mlh_q") mMlhQOutputName = name;
            else if (name == "mlh") mMlhOutputName = name;
        }
        if (mMlhOutputName.empty() && !scalarOutputs.empty())
        {
            mMlhOutputName = scalarOutputs.front();
            if (scalarOutputs.size() >= 2 && mMlhQOutputName.empty())
            {
                mMlhQOutputName = scalarOutputs[1];
            }
        }

        if (mInputName.empty() || mPolicyOutputName.empty() || mValueOutputName.empty())
        {
            throw std::runtime_error("Failed to determine small-model tensor names.");
        }
        // mlh / mlh_q outputs are optional: an older engine without either
        // works (the modifier collapses to zero so old engines see the same
        // selection behavior as before this feature shipped).
    }

    nvinfer1::ICudaEngine& engine() { return *mEngine; }
    const std::string& inputName() const { return mInputName; }
    const std::string& policyName() const { return mPolicyOutputName; }
    const std::string& valueName() const { return mValueOutputName; }
    const std::string& mlhName() const { return mMlhOutputName; }
    const std::string& mlhQName() const { return mMlhQOutputName; }
    bool hasMlh() const { return !mMlhOutputName.empty(); }
    bool hasMlhQ() const { return !mMlhQOutputName.empty(); }

private:
    std::string mEnginePath;
    Logger mLogger;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::string mInputName;
    std::string mPolicyOutputName;
    std::string mValueOutputName;
    std::string mMlhOutputName;
    std::string mMlhQOutputName;
};

SmallEngineContext& getSmallEngineContext(const std::string& engine_path)
{
    static auto* mutex = new std::mutex();
    static auto* contexts = new std::unordered_map<std::string, std::unique_ptr<SmallEngineContext>>();

    std::string resolved = resolve_engine_path(engine_path, "model.trt");
    std::lock_guard<std::mutex> lock(*mutex);
    auto it = contexts->find(resolved);
    if (it == contexts->end())
    {
        auto ctx = std::make_unique<SmallEngineContext>(resolved);
        auto [inserted, _] = contexts->emplace(resolved, std::move(ctx));
        return *inserted->second;
    }
    return *it->second;
}

size_t encodeSmallIndex(int row, int col, int plane)
{
    return ((static_cast<size_t>(row) * kSmallBoardSize + static_cast<size_t>(col)) * kSmallFeaturePlanes)
        + static_cast<size_t>(plane);
}

std::vector<float> encodeSmallFeatures(const chess::Board& board)
{
    std::vector<float> features(static_cast<size_t>(kSmallBoardSize) * kSmallBoardSize * kSmallFeaturePlanes, 0.0F);

    const std::array<std::pair<chess::PieceType, int>, 6> typeOrder{{
        {chess::PieceType::PAWN, 0},
        {chess::PieceType::KNIGHT, 1},
        {chess::PieceType::BISHOP, 2},
        {chess::PieceType::ROOK, 3},
        {chess::PieceType::QUEEN, 4},
        {chess::PieceType::KING, 5},
    }};

    for (auto color : {chess::Color::WHITE, chess::Color::BLACK})
    {
        const int base = color == chess::Color::WHITE ? 0 : 6;
        for (const auto& [pieceType, offset] : typeOrder)
        {
            chess::Bitboard bb = board.pieces(pieceType, color);
            while (bb)
            {
                const int squareIndex = static_cast<int>(bb.pop());
                const int rank = squareIndex / kSmallBoardSize;
                const int file = squareIndex % kSmallBoardSize;
                const int plane = base + offset;
                features[encodeSmallIndex(kSmallBoardSize - 1 - rank, file, plane)] = 1.0F;
            }
        }
    }

    auto fillPlane = [&](int plane, float value) {
        for (int r = 0; r < kSmallBoardSize; ++r)
        {
            for (int c = 0; c < kSmallBoardSize; ++c)
            {
                features[encodeSmallIndex(r, c, plane)] = value;
            }
        }
    };

    fillPlane(12, board.sideToMove() == chess::Color::WHITE ? 1.0F : 0.0F);

    const auto enPassant = board.enpassantSq();
    if (enPassant.is_valid())
    {
        const int idxSq = enPassant.index();
        const int rank = idxSq / kSmallBoardSize;
        const int file = idxSq % kSmallBoardSize;
        features[encodeSmallIndex(kSmallBoardSize - 1 - rank, file, 13)] = 1.0F;
    }
    else
    {
        fillPlane(13, 0.0F);
    }

    const auto castling = board.castlingRights();
    fillPlane(
        14,
        castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
    fillPlane(
        15,
        castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);
    fillPlane(
        16,
        castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
    fillPlane(
        17,
        castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);

    const float halfMoveNorm = std::min(1.0F, static_cast<float>(board.halfMoveClock()) / 100.0F);
    fillPlane(18, halfMoveNorm);

    return features;
}

SmallEncodedPosition build_small_encoded_position(const chess::Board& board)
{
    SmallEncodedPosition encoded;
    encoded.features = encodeSmallFeatures(board);
    return encoded;
}

SmallRawOutputs runSmallInference(
    SmallEngineContext& ctx,
    const std::vector<float>& input)
{
    const size_t kInputElements
        = static_cast<size_t>(kSmallBoardSize) * kSmallBoardSize * kSmallFeaturePlanes;
    const size_t kPolicyElements = static_cast<size_t>(kSmallPolicySize);
    const size_t kValueElements = static_cast<size_t>(kSmallValueSize);
    const size_t kMlhElements = 1;

    if (input.size() != kInputElements)
    {
        throw std::runtime_error("Small-model single-batch received unexpected feature vector size.");
    }

    nvinfer1::ICudaEngine& engine = ctx.engine();

    std::unique_ptr<nvinfer1::IExecutionContext> exec(engine.createExecutionContext());
    if (!exec)
    {
        throw std::runtime_error("Failed to create small-model execution context.");
    }

    const auto requestedInput = nvinfer1::Dims4{1, kSmallBoardSize, kSmallBoardSize, kSmallFeaturePlanes};
    if (!exec->setInputShape(ctx.inputName().c_str(), requestedInput))
    {
        throw std::runtime_error("Failed to set small-model input shape.");
    }

    const auto inputType = engine.getTensorDataType(ctx.inputName().c_str());
    const auto policyType = engine.getTensorDataType(ctx.policyName().c_str());
    const auto valueType = engine.getTensorDataType(ctx.valueName().c_str());
    const bool hasMlh = ctx.hasMlh();
    const bool hasMlhQ = ctx.hasMlhQ();
    const auto mlhType = hasMlh ? engine.getTensorDataType(ctx.mlhName().c_str()) : nvinfer1::DataType::kFLOAT;
    const auto mlhQType = hasMlhQ ? engine.getTensorDataType(ctx.mlhQName().c_str()) : nvinfer1::DataType::kFLOAT;

    DeviceBuffer inputDevice(kInputElements * element_size(inputType));
    DeviceBuffer policyDevice(kPolicyElements * element_size(policyType));
    DeviceBuffer valueDevice(kValueElements * element_size(valueType));
    std::unique_ptr<DeviceBuffer> mlhDevice;
    std::unique_ptr<DeviceBuffer> mlhQDevice;
    if (hasMlh) mlhDevice = std::make_unique<DeviceBuffer>(kMlhElements * element_size(mlhType));
    if (hasMlhQ) mlhQDevice = std::make_unique<DeviceBuffer>(kMlhElements * element_size(mlhQType));
    CudaStream stream;

    std::vector<__half> inputHalf;
    const void* inputHost = nullptr;
    size_t inputBytes = 0;

    if (inputType == nvinfer1::DataType::kFLOAT)
    {
        inputHost = input.data();
        inputBytes = kInputElements * sizeof(float);
    }
    else if (inputType == nvinfer1::DataType::kHALF)
    {
        inputHalf.resize(kInputElements);
        for (size_t i = 0; i < kInputElements; ++i)
        {
            inputHalf[i] = __float2half(input[i]);
        }
        inputHost = inputHalf.data();
        inputBytes = inputHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported small-model input tensor type.");
    }

    if (cudaMemcpyAsync(inputDevice.data(), inputHost, inputBytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (small single H2D) failed.");
    }

    if (!exec->setTensorAddress(ctx.inputName().c_str(), inputDevice.data())
        || !exec->setTensorAddress(ctx.policyName().c_str(), policyDevice.data())
        || !exec->setTensorAddress(ctx.valueName().c_str(), valueDevice.data()))
    {
        throw std::runtime_error("Failed to set small-model tensor addresses.");
    }
    if (hasMlh && !exec->setTensorAddress(ctx.mlhName().c_str(), mlhDevice->data()))
    {
        throw std::runtime_error("Failed to set small-model mlh tensor address.");
    }
    if (hasMlhQ && !exec->setTensorAddress(ctx.mlhQName().c_str(), mlhQDevice->data()))
    {
        throw std::runtime_error("Failed to set small-model mlh_q tensor address.");
    }

    if (!exec->enqueueV3(stream.get()))
    {
        throw std::runtime_error("Small-model single-batch TensorRT enqueue failed.");
    }

    SmallRawOutputs outputs;
    outputs.policy.resize(kPolicyElements);
    outputs.wdl.resize(kValueElements);
    outputs.mlh.assign(kMlhElements, 0.0F);
    outputs.mlh_q.assign(kMlhElements, 0.0F);

    std::vector<__half> policyHalf;
    std::vector<__half> valueHalf;
    std::vector<__half> mlhHalf;
    std::vector<__half> mlhQHalf;

    void* policyHost = nullptr;
    size_t policyBytes = 0;

    if (policyType == nvinfer1::DataType::kFLOAT)
    {
        policyHost = outputs.policy.data();
        policyBytes = outputs.policy.size() * sizeof(float);
    }
    else if (policyType == nvinfer1::DataType::kHALF)
    {
        policyHalf.resize(outputs.policy.size());
        policyHost = policyHalf.data();
        policyBytes = policyHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported small-model policy tensor type.");
    }

    void* valueHost = nullptr;
    size_t valueBytes = 0;

    if (valueType == nvinfer1::DataType::kFLOAT)
    {
        valueHost = outputs.wdl.data();
        valueBytes = outputs.wdl.size() * sizeof(float);
    }
    else if (valueType == nvinfer1::DataType::kHALF)
    {
        valueHalf.resize(outputs.wdl.size());
        valueHost = valueHalf.data();
        valueBytes = valueHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported small-model value tensor type.");
    }

    if (cudaMemcpyAsync(policyHost, policyDevice.data(), policyBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (small single policy D2H) failed.");
    }

    if (cudaMemcpyAsync(valueHost, valueDevice.data(), valueBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (small single value D2H) failed.");
    }

    void* mlhHost = nullptr;
    size_t mlhBytes = 0;
    if (hasMlh)
    {
        if (mlhType == nvinfer1::DataType::kFLOAT)
        {
            mlhHost = outputs.mlh.data();
            mlhBytes = outputs.mlh.size() * sizeof(float);
        }
        else if (mlhType == nvinfer1::DataType::kHALF)
        {
            mlhHalf.resize(outputs.mlh.size());
            mlhHost = mlhHalf.data();
            mlhBytes = mlhHalf.size() * sizeof(__half);
        }
        else
        {
            throw std::runtime_error("Unsupported small-model mlh tensor type.");
        }
        if (cudaMemcpyAsync(mlhHost, mlhDevice->data(), mlhBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpyAsync (small single mlh D2H) failed.");
        }
    }

    void* mlhQHost = nullptr;
    size_t mlhQBytes = 0;
    if (hasMlhQ)
    {
        if (mlhQType == nvinfer1::DataType::kFLOAT)
        {
            mlhQHost = outputs.mlh_q.data();
            mlhQBytes = outputs.mlh_q.size() * sizeof(float);
        }
        else if (mlhQType == nvinfer1::DataType::kHALF)
        {
            mlhQHalf.resize(outputs.mlh_q.size());
            mlhQHost = mlhQHalf.data();
            mlhQBytes = mlhQHalf.size() * sizeof(__half);
        }
        else
        {
            throw std::runtime_error("Unsupported small-model mlh_q tensor type.");
        }
        if (cudaMemcpyAsync(mlhQHost, mlhQDevice->data(), mlhQBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpyAsync (small single mlh_q D2H) failed.");
        }
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaStreamSynchronize (small single) failed.");
    }

    if (policyType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.policy.size(); ++i)
        {
            outputs.policy[i] = __half2float(policyHalf[i]);
        }
    }

    if (valueType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.wdl.size(); ++i)
        {
            outputs.wdl[i] = __half2float(valueHalf[i]);
        }
    }

    if (hasMlh && mlhType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.mlh.size(); ++i)
        {
            outputs.mlh[i] = __half2float(mlhHalf[i]);
        }
    }
    if (hasMlhQ && mlhQType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.mlh_q.size(); ++i)
        {
            outputs.mlh_q[i] = __half2float(mlhQHalf[i]);
        }
    }

    return outputs;
}

SmallPolicyValue evaluateSmall(
    std::string_view fen,
    const std::string& engine_path)
{
    const std::string fenString = fen.empty() ? std::string(chess::constants::STARTPOS) : std::string(fen);
    chess::Board board(fenString);

    SmallEncodedPosition encoded = build_small_encoded_position(board);
    SmallEngineContext& ctx = getSmallEngineContext(engine_path);
    SmallRawOutputs raw = runSmallInference(ctx, encoded.features);

    if (raw.policy.size() != static_cast<size_t>(kSmallPolicySize))
    {
        throw std::runtime_error("Unexpected policy size from small single-batch inference.");
    }
    if (raw.wdl.size() != static_cast<size_t>(kSmallValueSize))
    {
        throw std::runtime_error("Unexpected value size from small single-batch inference.");
    }

    SmallPolicyValue result;
    result.policy_logits = std::move(raw.policy);
    for (int i = 0; i < kSmallValueSize; ++i)
    {
        result.wdl[static_cast<size_t>(i)] = raw.wdl[static_cast<size_t>(i)];
    }
    result.mlh = raw.mlh.empty() ? 0.0F : raw.mlh[0];
    // Fall back to mlh when mlh_q isn't exported (legacy single-head engine).
    result.mlh_q = !raw.mlh_q.empty() ? raw.mlh_q[0]
                  : (raw.mlh.empty() ? 0.0F : raw.mlh[0]);
    return result;
}


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

SmallEncodedPosition encode_small_position(const chess::Board& board)
{
    return build_small_encoded_position(board);
}

struct SmallBatchRunner::Impl
{
    struct SmallPending
    {
        SmallEncodedPosition position;
        std::shared_ptr<std::promise<SmallPolicyValue>> promise;
    };

    // Per-worker inference state. Each worker owns its own IExecutionContext +
    // CUDA stream + device/host buffers so they can pipeline: while worker N is
    // busy on the GPU, worker N+1 can be prepping its next batch in parallel.
    // The MCTS-side queue (queue/mutex/cv) is shared across workers, so each
    // worker pops a sub-batch independently and serves it without blocking
    // others. TensorRT explicitly supports concurrent execution of multiple
    // IExecutionContext instances on the same engine.
    struct Worker
    {
        std::thread thread;
        std::unique_ptr<nvinfer1::IExecutionContext> exec;
        std::unique_ptr<CudaStream> stream;
        ResizableDeviceBuffer inputDevice;
        ResizableDeviceBuffer policyDevice;
        ResizableDeviceBuffer valueDevice;
        ResizableDeviceBuffer mlhDevice;
        ResizableDeviceBuffer mlhQDevice;
        PinnedBuffer hostInput;
        PinnedBuffer hostPolicy;
        PinnedBuffer hostValue;
        PinnedBuffer hostMlh;
        PinnedBuffer hostMlhQ;
    };

    explicit Impl(SmallBatchConfig cfg)
        : config(std::move(cfg))
        , ctx(getSmallEngineContext(config.engine_path))
        , flush_threshold_(static_cast<size_t>(config.max_batch_size))
    {
        if (config.max_batch_size <= 0)
        {
            throw std::invalid_argument("SmallBatchRunner requires positive max_batch_size");
        }

        // Number of concurrent inference workers. Override via
        // CHESSRL_INFER_WORKERS. Each adds an IExecutionContext + CUDA stream
        // + persistent device/host buffers; while worker N is on the GPU,
        // worker N+1 is prepping H2D for its batch.
        //   measured (H200, threads=256, sims=800, b=1024): peak at W=4
        //   (~262 moves/s, vs ~117 with W=1, threads=64). W=6 starts to
        //   regress; W=8 cliffs hard from queue + stream contention.
        // On smaller GPUs (e.g. a 4090) W=2 is likely the right default —
        // override the env var if so.
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

    BatchTicket<SmallPolicyValue> enqueue(SmallEncodedPosition position)
    {
        const size_t expected = static_cast<size_t>(kSmallBoardSize) * kSmallBoardSize * kSmallFeaturePlanes;
        if (position.features.size() != expected)
        {
            throw std::runtime_error("Small batch request provided incorrect feature vector length.");
        }

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

        const auto ep_guard = [&work](const std::exception_ptr& ep) {
            fulfillWithException(work, ep);
        };

        try
        {
            auto start_time = std::chrono::steady_clock::now();
            if (!w.exec)
            {
                w.exec.reset(ctx.engine().createExecutionContext());
                if (!w.exec)
                {
                    throw std::runtime_error("Failed to create small-model batch execution context.");
                }
            }
            if (!w.stream)
            {
                w.stream = std::make_unique<CudaStream>();
            }

            auto& engine = ctx.engine();
            const auto inputType = engine.getTensorDataType(ctx.inputName().c_str());
            const auto policyType = engine.getTensorDataType(ctx.policyName().c_str());
            const auto valueType = engine.getTensorDataType(ctx.valueName().c_str());
            const bool hasMlh = ctx.hasMlh();
            const bool hasMlhQ = ctx.hasMlhQ();
            const auto mlhType = hasMlh
                ? engine.getTensorDataType(ctx.mlhName().c_str())
                : nvinfer1::DataType::kFLOAT;
            const auto mlhQType = hasMlhQ
                ? engine.getTensorDataType(ctx.mlhQName().c_str())
                : nvinfer1::DataType::kFLOAT;

            const size_t actualCount = work.size();
            const size_t inputElements
                = static_cast<size_t>(kSmallBoardSize) * kSmallBoardSize * kSmallFeaturePlanes;
            const size_t policyElements = static_cast<size_t>(kSmallPolicySize);
            const size_t valueElements = static_cast<size_t>(kSmallValueSize);
            const size_t mlhElements = 1;

            nvinfer1::Dims inputDims = engine.getTensorShape(ctx.inputName().c_str());
            if (inputDims.nbDims <= 0)
            {
                throw std::runtime_error("Small-model input tensor has unexpected rank.");
            }
            
            // Step A: Enforce dynamic dimensions - throw error if engine is static
            if (inputDims.d[0] != -1)
            {
                throw std::runtime_error("Loaded engine is static! Please generate a dynamic engine with trtexec --minShapes/--optShapes/--maxShapes.");
            }
            
            // Step B: Remove padding logic - use exact batch size
            const size_t engineBatch = actualCount;
            inputDims.d[0] = static_cast<int>(engineBatch);
            if (!w.exec->setInputShape(ctx.inputName().c_str(), inputDims))
            {
                throw std::runtime_error("Failed to set small-model batch input shape.");
            }

            // Step 2: Use persistent buffers (per-worker)
            w.inputDevice.ensure(engineBatch * inputElements * element_size(inputType));
            w.policyDevice.ensure(engineBatch * policyElements * element_size(policyType));
            w.valueDevice.ensure(engineBatch * valueElements * element_size(valueType));
            const size_t mlhBytes = hasMlh
                ? engineBatch * mlhElements * element_size(mlhType)
                : 0;
            const size_t mlhQBytes = hasMlhQ
                ? engineBatch * mlhElements * element_size(mlhQType)
                : 0;
            if (hasMlh) w.mlhDevice.ensure(mlhBytes);
            if (hasMlhQ) w.mlhQDevice.ensure(mlhQBytes);

            // Step 2: Ensure pinned host buffers (per-worker)
            const size_t inputBytes = engineBatch * inputElements * element_size(inputType);
            const size_t policyBytes = engineBatch * policyElements * element_size(policyType);
            const size_t valueBytes = engineBatch * valueElements * element_size(valueType);
            w.hostInput.ensure(inputBytes);
            w.hostPolicy.ensure(policyBytes);
            w.hostValue.ensure(valueBytes);
            if (hasMlh) w.hostMlh.ensure(mlhBytes);
            if (hasMlhQ) w.hostMlhQ.ensure(mlhQBytes);

            // Step 3: Input assembly - write directly into pinned buffer
            // Step C: Simplified copy loops - no padding needed since engineBatch == actualCount
            float* hostInputFloat = static_cast<float*>(w.hostInput.data());
            __half* hostInputHalf = static_cast<__half*>(w.hostInput.data());
            
            for (size_t i = 0; i < actualCount; ++i)
            {
                const auto& features = work[i].position.features;
                const size_t offset = i * inputElements;
                
                if (inputType == nvinfer1::DataType::kFLOAT)
                {
                    std::memcpy(hostInputFloat + offset, features.data(), inputElements * sizeof(float));
                }
                else if (inputType == nvinfer1::DataType::kHALF)
                {
                    // Convert directly into pinned buffer
                    for (size_t j = 0; j < inputElements; ++j)
                    {
                        hostInputHalf[offset + j] = __float2half(features[j]);
                    }
                }
                else
                {
                    throw std::runtime_error("Unsupported small-model batch input tensor type.");
                }
            }
            
            if (cudaMemcpyAsync(w.inputDevice.data(), w.hostInput.data(), inputBytes, cudaMemcpyHostToDevice, w.stream->get()) != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyAsync (small batch H2D) failed.");
            }

            if (!w.exec->setTensorAddress(ctx.inputName().c_str(), w.inputDevice.data())
                || !w.exec->setTensorAddress(ctx.policyName().c_str(), w.policyDevice.data())
                || !w.exec->setTensorAddress(ctx.valueName().c_str(), w.valueDevice.data()))
            {
                throw std::runtime_error("Failed to set small-model batch tensor addresses.");
            }
            if (hasMlh && !w.exec->setTensorAddress(ctx.mlhName().c_str(), w.mlhDevice.data()))
            {
                throw std::runtime_error("Failed to set small-model batch mlh tensor address.");
            }
            if (hasMlhQ && !w.exec->setTensorAddress(ctx.mlhQName().c_str(), w.mlhQDevice.data()))
            {
                throw std::runtime_error("Failed to set small-model batch mlh_q tensor address.");
            }

            if (!w.exec->enqueueV3(w.stream->get()))
            {
                throw std::runtime_error("Small-model batched TensorRT enqueue failed.");
            }

            // Step 3: D2H transfer - copy directly into pinned buffers
            if (cudaMemcpyAsync(w.hostPolicy.data(), w.policyDevice.data(), policyBytes, cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyAsync (small batch policy D2H) failed.");
            }
            if (cudaMemcpyAsync(w.hostValue.data(), w.valueDevice.data(), valueBytes, cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyAsync (small batch value D2H) failed.");
            }
            if (hasMlh && cudaMemcpyAsync(w.hostMlh.data(), w.mlhDevice.data(), mlhBytes, cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyAsync (small batch mlh D2H) failed.");
            }
            if (hasMlhQ && cudaMemcpyAsync(w.hostMlhQ.data(), w.mlhQDevice.data(), mlhQBytes, cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyAsync (small batch mlh_q D2H) failed.");
            }

            if (cudaStreamSynchronize(w.stream->get()) != cudaSuccess)
            {
                throw std::runtime_error("cudaStreamSynchronize (small batch) failed.");
            }

            if (telemetry_enabled_)
            {
                const double infer_ms = std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - start_time)
                                              .count();
                std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
                batches_ += 1;
                positions_ += actualCount;
                total_infer_ms_ += infer_ms;
            }

            // Step 3: Distribution - read directly from pinned buffers
            float* policyFloat = static_cast<float*>(w.hostPolicy.data());
            __half* policyHalf = static_cast<__half*>(w.hostPolicy.data());
            float* valueFloat = static_cast<float*>(w.hostValue.data());
            __half* valueHalf = static_cast<__half*>(w.hostValue.data());
            float* mlhFloat = hasMlh ? static_cast<float*>(w.hostMlh.data()) : nullptr;
            __half* mlhHalf = hasMlh ? static_cast<__half*>(w.hostMlh.data()) : nullptr;
            float* mlhQFloat = hasMlhQ ? static_cast<float*>(w.hostMlhQ.data()) : nullptr;
            __half* mlhQHalf = hasMlhQ ? static_cast<__half*>(w.hostMlhQ.data()) : nullptr;

            for (size_t i = 0; i < actualCount; ++i)
            {
                SmallPolicyValue result;
                const size_t policyOffset = i * policyElements;
                const size_t valueOffset = i * valueElements;
                
                // Read policy logits
                result.policy_logits.reserve(policyElements);
                if (policyType == nvinfer1::DataType::kFLOAT)
                {
                    result.policy_logits.assign(
                        policyFloat + policyOffset,
                        policyFloat + policyOffset + policyElements);
                }
                else if (policyType == nvinfer1::DataType::kHALF)
                {
                    result.policy_logits.resize(policyElements);
                    for (size_t j = 0; j < policyElements; ++j)
                    {
                        result.policy_logits[j] = __half2float(policyHalf[policyOffset + j]);
                    }
                }
                else
                {
                    throw std::runtime_error("Unsupported small-model batch policy tensor type.");
                }
                
                // Read value/WDL
                if (valueType == nvinfer1::DataType::kFLOAT)
                {
                    for (size_t j = 0; j < valueElements; ++j)
                    {
                        result.wdl[j] = valueFloat[valueOffset + j];
                    }
                }
                else if (valueType == nvinfer1::DataType::kHALF)
                {
                    for (size_t j = 0; j < valueElements; ++j)
                    {
                        result.wdl[j] = __half2float(valueHalf[valueOffset + j]);
                    }
                }
                else
                {
                    throw std::runtime_error("Unsupported small-model batch value tensor type.");
                }

                // Read MLH heads (single scalar per row each). When the engine
                // has no mlh / mlh_q head, leave the default 0 — the modifier
                // collapses to zero so old engines see the same selection
                // behavior as before this feature shipped.
                if (hasMlh)
                {
                    const size_t mlhOffset = i * mlhElements;
                    if (mlhType == nvinfer1::DataType::kFLOAT)
                    {
                        result.mlh = mlhFloat[mlhOffset];
                    }
                    else if (mlhType == nvinfer1::DataType::kHALF)
                    {
                        result.mlh = __half2float(mlhHalf[mlhOffset]);
                    }
                    else
                    {
                        throw std::runtime_error("Unsupported small-model batch mlh tensor type.");
                    }
                }
                if (hasMlhQ)
                {
                    const size_t mlhOffset = i * mlhElements;
                    if (mlhQType == nvinfer1::DataType::kFLOAT)
                    {
                        result.mlh_q = mlhQFloat[mlhOffset];
                    }
                    else if (mlhQType == nvinfer1::DataType::kHALF)
                    {
                        result.mlh_q = __half2float(mlhQHalf[mlhOffset]);
                    }
                    else
                    {
                        throw std::runtime_error("Unsupported small-model batch mlh_q tensor type.");
                    }
                }
                else
                {
                    // Legacy single-head fallback: mlh stands in for mlh_q so
                    // the MCTS leaf seed is at least the model's true-target
                    // estimate rather than zero.
                    result.mlh_q = result.mlh;
                }

                work[i].promise->set_value(std::move(result));
            }
        }
        catch (...)
        {
            ep_guard(std::current_exception());
            throw;
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
    SmallEngineContext& ctx;
    size_t flush_threshold_{0};
    // Per-worker IExecutionContext + CUDA stream + buffers. See Worker struct.
    // Stable storage via unique_ptr so worker threads can hold raw references.
    std::vector<std::unique_ptr<Worker>> workers_;
    mutable std::mutex mutex;
    std::condition_variable cv;
    bool stop_worker_{false};
    std::deque<SmallPending> queue;
    // Smaller flush timeout = less GPU idle when the queue is partially filled.
    // 100us is well under typical batch inference latency for the 1.2M-param model
    // at any reasonable batch size, so this is mostly free latency reclaimed.
    // Override via CHESSRL_MCTS_FLUSH_US if needed.
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
        oss << resolve_engine_path(config.engine_path, "model.trt")
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

SmallBatchRunner::Ticket SmallBatchRunner::enqueue(SmallEncodedPosition position)
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

