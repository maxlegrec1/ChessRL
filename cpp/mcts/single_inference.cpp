#include "mcts/single_inference.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
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

namespace chessrl::mcts::single
{
namespace
{

constexpr int kSmallBoardSize = 8;
constexpr int kSmallFeaturePlanes = 19;
constexpr int kSmallPolicySize = 1929;
constexpr int kSmallValueSize = 3;

constexpr float kIllegalScore = -1e9F;

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

std::string resolveEnginePath(const std::string& engine_path, const std::string& default_name)
{
    std::string resolved = engine_path.empty() ? default_name : engine_path;
    try
    {
        resolved = std::filesystem::absolute(resolved).string();
    }
    catch (...)
    {
    }
    return resolved;
}

size_t getElementSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        throw std::runtime_error("Unsupported TensorRT data type");
    }
}

size_t volume(const nvinfer1::Dims& dims)
{
    size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        result *= static_cast<size_t>(dims.d[i]);
    }
    return result;
}

size_t normalized_tensor_volume(const nvinfer1::Dims& dims)
{
    const size_t total = volume(dims);
    if (dims.nbDims > 0)
    {
        const int first = dims.d[0];
        if (first > 0)
        {
            const size_t batch = static_cast<size_t>(first);
            if (batch > 0 && total % batch == 0)
            {
                return total / batch;
            }
        }
    }
    return total;
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
            }
        }

        if (mInputName.empty() || mPolicyOutputName.empty() || mValueOutputName.empty())
        {
            throw std::runtime_error("Failed to determine small-model tensor names.");
        }
    }

    nvinfer1::ICudaEngine& engine() { return *mEngine; }
    const std::string& inputName() const { return mInputName; }
    const std::string& policyName() const { return mPolicyOutputName; }
    const std::string& valueName() const { return mValueOutputName; }

private:
    std::string mEnginePath;
    Logger mLogger;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::string mInputName;
    std::string mPolicyOutputName;
    std::string mValueOutputName;
};

SmallEngineContext& getSmallEngineContext(const std::string& engine_path)
{
    static auto* mutex = new std::mutex();
    static auto* contexts = new std::unordered_map<std::string, std::unique_ptr<SmallEngineContext>>();

    std::string resolved = resolveEnginePath(engine_path, "model.trt");
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

    DeviceBuffer inputDevice(kInputElements * getElementSize(inputType));
    DeviceBuffer policyDevice(kPolicyElements * getElementSize(policyType));
    DeviceBuffer valueDevice(kValueElements * getElementSize(valueType));
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

    if (!exec->enqueueV3(stream.get()))
    {
        throw std::runtime_error("Small-model single-batch TensorRT enqueue failed.");
    }

    SmallRawOutputs outputs;
    outputs.policy.resize(kPolicyElements);
    outputs.wdl.resize(kValueElements);

    std::vector<__half> policyHalf;
    std::vector<__half> valueHalf;

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

    explicit Impl(SmallBatchConfig cfg)
        : config(std::move(cfg))
        , ctx(getSmallEngineContext(config.engine_path))
        , flush_threshold_(static_cast<size_t>(config.max_batch_size))
    {
        if (config.max_batch_size <= 0)
        {
            throw std::invalid_argument("SmallBatchRunner requires positive max_batch_size");
        }
        worker = std::thread([this]() { worker_loop(); });
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop_worker_ = true;
        }
        cv.notify_all();
        if (worker.joinable())
        {
            worker.join();
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

    void process()
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
            if (!exec)
            {
                exec.reset(ctx.engine().createExecutionContext());
                if (!exec)
                {
                    throw std::runtime_error("Failed to create small-model batch execution context.");
                }
            }
            if (!stream)
            {
                stream = std::make_unique<CudaStream>();
            }

            auto& engine = ctx.engine();
            const auto inputType = engine.getTensorDataType(ctx.inputName().c_str());
            const auto policyType = engine.getTensorDataType(ctx.policyName().c_str());
            const auto valueType = engine.getTensorDataType(ctx.valueName().c_str());

            const size_t actualCount = work.size();
            const size_t inputElements
                = static_cast<size_t>(kSmallBoardSize) * kSmallBoardSize * kSmallFeaturePlanes;
            const size_t policyElements = static_cast<size_t>(kSmallPolicySize);
            const size_t valueElements = static_cast<size_t>(kSmallValueSize);

            nvinfer1::Dims inputDims = engine.getTensorShape(ctx.inputName().c_str());
            if (inputDims.nbDims <= 0)
            {
                throw std::runtime_error("Small-model input tensor has unexpected rank.");
            }
            if (inputDims.d[0] == -1)
            {
                inputDims.d[0] = static_cast<int>(actualCount);
            }
            const size_t engineBatch = inputDims.d[0] == -1 ? actualCount : static_cast<size_t>(inputDims.d[0]);
            if (engineBatch < actualCount)
            {
                throw std::runtime_error("Small-model engine batch is smaller than requested work size.");
            }
            inputDims.d[0] = static_cast<int>(engineBatch);
            if (!exec->setInputShape(ctx.inputName().c_str(), inputDims))
            {
                throw std::runtime_error("Failed to set small-model batch input shape.");
            }

            inputDevice.ensure(engineBatch * inputElements * getElementSize(inputType));
            policyDevice.ensure(engineBatch * policyElements * getElementSize(policyType));
            valueDevice.ensure(engineBatch * valueElements * getElementSize(valueType));

            std::vector<float> concatenated(engineBatch * inputElements, 0.0F);
            for (size_t i = 0; i < engineBatch; ++i)
            {
                const auto& features = (i < actualCount) ? work[i].position.features : work.back().position.features;
                std::copy(features.begin(), features.end(), concatenated.begin() + static_cast<ptrdiff_t>(i * inputElements));
            }

            std::vector<__half> inputHalf;
            const void* inputHost = nullptr;
            size_t inputBytes = 0;
            if (inputType == nvinfer1::DataType::kFLOAT)
            {
                inputHost = concatenated.data();
                inputBytes = concatenated.size() * sizeof(float);
            }
            else if (inputType == nvinfer1::DataType::kHALF)
            {
                inputHalf.resize(concatenated.size());
                for (size_t i = 0; i < concatenated.size(); ++i)
                {
                    inputHalf[i] = __float2half(concatenated[i]);
                }
                inputHost = inputHalf.data();
                inputBytes = inputHalf.size() * sizeof(__half);
            }
            else
            {
                throw std::runtime_error("Unsupported small-model batch input tensor type.");
            }

            if (cudaMemcpyAsync(inputDevice.data(), inputHost, inputBytes, cudaMemcpyHostToDevice, stream->get()) != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyAsync (small batch H2D) failed.");
            }

            if (!exec->setTensorAddress(ctx.inputName().c_str(), inputDevice.data())
                || !exec->setTensorAddress(ctx.policyName().c_str(), policyDevice.data())
                || !exec->setTensorAddress(ctx.valueName().c_str(), valueDevice.data()))
            {
                throw std::runtime_error("Failed to set small-model batch tensor addresses.");
            }

            if (!exec->enqueueV3(stream->get()))
            {
                throw std::runtime_error("Small-model batched TensorRT enqueue failed.");
            }

            const size_t totalPolicy = engineBatch * policyElements;
            const size_t totalValue = engineBatch * valueElements;

            std::vector<float> policyFloat(totalPolicy);
            std::vector<float> valueFloat(totalValue);
            std::vector<__half> policyHalf;
            std::vector<__half> valueHalf;

            void* policyHost = nullptr;
            size_t policyBytes = 0;
            if (policyType == nvinfer1::DataType::kFLOAT)
            {
                policyHost = policyFloat.data();
                policyBytes = policyFloat.size() * sizeof(float);
            }
            else if (policyType == nvinfer1::DataType::kHALF)
            {
                policyHalf.resize(totalPolicy);
                policyHost = policyHalf.data();
                policyBytes = policyHalf.size() * sizeof(__half);
            }
            else
            {
                throw std::runtime_error("Unsupported small-model batch policy tensor type.");
            }

            void* valueHost = nullptr;
            size_t valueBytes = 0;
            if (valueType == nvinfer1::DataType::kFLOAT)
            {
                valueHost = valueFloat.data();
                valueBytes = valueFloat.size() * sizeof(float);
            }
            else if (valueType == nvinfer1::DataType::kHALF)
            {
                valueHalf.resize(totalValue);
                valueHost = valueHalf.data();
                valueBytes = valueHalf.size() * sizeof(__half);
            }
            else
            {
                throw std::runtime_error("Unsupported small-model batch value tensor type.");
            }

            if (cudaMemcpyAsync(policyHost, policyDevice.data(), policyBytes, cudaMemcpyDeviceToHost, stream->get()) != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyAsync (small batch policy D2H) failed.");
            }
            if (cudaMemcpyAsync(valueHost, valueDevice.data(), valueBytes, cudaMemcpyDeviceToHost, stream->get()) != cudaSuccess)
            {
                throw std::runtime_error("cudaMemcpyAsync (small batch value D2H) failed.");
            }

            if (cudaStreamSynchronize(stream->get()) != cudaSuccess)
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

            if (policyType == nvinfer1::DataType::kHALF)
            {
                policyFloat.resize(totalPolicy);
                for (size_t i = 0; i < totalPolicy; ++i)
                {
                    policyFloat[i] = __half2float(policyHalf[i]);
                }
            }
            if (valueType == nvinfer1::DataType::kHALF)
            {
                valueFloat.resize(totalValue);
                for (size_t i = 0; i < totalValue; ++i)
                {
                    valueFloat[i] = __half2float(valueHalf[i]);
                }
            }

            for (size_t i = 0; i < actualCount; ++i)
            {
                SmallPolicyValue result;
                result.policy_logits.assign(
                    policyFloat.begin() + static_cast<ptrdiff_t>(i * policyElements),
                    policyFloat.begin() + static_cast<ptrdiff_t>((i + 1) * policyElements));
                for (size_t j = 0; j < valueElements; ++j)
                {
                    result.wdl[j] = valueFloat[i * valueElements + j];
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

    void worker_loop()
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_for(lock, flush_timeout_, [this]() { return stop_worker_ || queue.size() >= flush_threshold_; });
            if (stop_worker_ && queue.empty())
            {
                break;
            }
            if (queue.empty())
            {
                continue;
            }
            lock.unlock();
            try
            {
                process();
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
    std::unique_ptr<nvinfer1::IExecutionContext> exec;
    std::unique_ptr<CudaStream> stream;
    ResizableDeviceBuffer inputDevice;
    ResizableDeviceBuffer policyDevice;
    ResizableDeviceBuffer valueDevice;
    mutable std::mutex mutex;
    std::condition_variable cv;
    bool stop_worker_{false};
    std::deque<SmallPending> queue;
    std::thread worker;
    const std::chrono::milliseconds flush_timeout_{1};
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
        oss << resolveEnginePath(config.engine_path, "model.trt")
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
    impl_->process();
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

