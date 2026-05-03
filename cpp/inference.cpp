#include "inference.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cctype>

#include "chess-library/include/chess.hpp"
#include "cuda_fp16.h"
#include "include/trt_utils.hpp"

namespace chessrl
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
constexpr int kSmallBatchSize = 1024;
constexpr int kSmallValueSize = 3;

constexpr float kIllegalScore = -1e9F;

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

namespace small_impl
{

struct RawOutputs
{
    std::vector<float> policy;
    std::vector<float> wdl;
};

class EngineContext
{
public:
    explicit EngineContext(std::string path)
        : enginePath(std::move(path))
    {
        std::ifstream engineFile(enginePath, std::ios::binary);
        if (!engineFile)
        {
            throw std::runtime_error(std::string("Failed to open engine file: ") + enginePath);
        }

        engineFile.seekg(0, std::ifstream::end);
        const auto fsize = engineFile.tellg();
        engineFile.seekg(0, std::ifstream::beg);

        std::vector<char> engineData(static_cast<size_t>(fsize));
        engineFile.read(engineData.data(), fsize);

        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime)
        {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine)
        {
            throw std::runtime_error("Failed to deserialize TensorRT engine.");
        }

        const int nbTensors = engine->getNbIOTensors();
        std::vector<std::string> outputNames;
        outputNames.reserve(nbTensors);

        for (int i = 0; i < nbTensors; ++i)
        {
            const char* name = engine->getIOTensorName(i);
            const auto mode = engine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                inputName = name;
            }
            else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
            {
                outputNames.emplace_back(name);
            }
        }

        for (const auto& name : outputNames)
        {
            const auto dims = engine->getTensorShape(name.c_str());
            const size_t total = volume(dims);
            if (total == static_cast<size_t>(kSmallBatchSize) * kSmallPolicySize && policyOutputName.empty())
            {
                policyOutputName = name;
            }
            else if (total == static_cast<size_t>(kSmallBatchSize) * kSmallValueSize && valueOutputName.empty())
            {
                valueOutputName = name;
            }
        }

        if (policyOutputName.empty() && !outputNames.empty())
        {
            policyOutputName = outputNames.front();
        }
        if (valueOutputName.empty() && outputNames.size() >= 2)
        {
            valueOutputName = outputNames[1];
        }

        if (inputName.empty() || policyOutputName.empty() || valueOutputName.empty())
        {
            throw std::runtime_error("Failed to determine input/output tensor names for small model.");
        }
    }

    std::string enginePath;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::string inputName;
    std::string policyOutputName;
    std::string valueOutputName;
};

EngineContext& getEngineContext(const std::string& engine_path)
{
    static auto* mutex = new std::mutex();
    static auto* contexts = new std::unordered_map<std::string, std::unique_ptr<EngineContext>>();

    std::string resolvedPath = resolve_engine_path(engine_path, "model.trt");

    std::lock_guard<std::mutex> lock(*mutex);
    auto it = contexts->find(resolvedPath);
    if (it == contexts->end())
    {
        auto ctx = std::make_unique<EngineContext>(resolvedPath);
        auto [inserted, _] = contexts->emplace(resolvedPath, std::move(ctx));
        return *inserted->second;
    }
    return *it->second;
}

class PolicyVocab
{
public:
    PolicyVocab()
    {
        std::ifstream file(kVocabPath);
        if (!file)
        {
            throw std::runtime_error(std::string("Failed to open policy vocab file: ") + kVocabPath);
        }

        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            tokens.emplace_back(line);
            if (!line.empty())
            {
                toIndex[line] = static_cast<int>(tokens.size() - 1);
            }
        }

        if (tokens.size() != kSmallPolicySize)
        {
            throw std::runtime_error("Unexpected small policy vocab size: " + std::to_string(tokens.size()));
        }
    }

    const std::string& at(int idx) const { return tokens.at(static_cast<size_t>(idx)); }

    std::optional<int> indexOf(const std::string& token) const
    {
        auto it = toIndex.find(token);
        if (it == toIndex.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    const std::vector<std::string>& tokensRef() const { return tokens; }

private:
    static constexpr const char* kVocabPath = "cpp/policy_index.txt";
    std::vector<std::string> tokens;
    std::unordered_map<std::string, int> toIndex;
};

PolicyVocab& getPolicyVocab()
{
    static PolicyVocab vocab;
    return vocab;
}

size_t encodeIndex(int batch, int row, int col, int plane)
{
    return (((static_cast<size_t>(batch) * kSmallBoardSize + static_cast<size_t>(row)) * kSmallBoardSize
             + static_cast<size_t>(col))
            * kSmallFeaturePlanes)
        + static_cast<size_t>(plane);
}

struct EncodedBatch
{
    std::vector<float> features;
    std::vector<chess::Board> boards;
};

EncodedBatch encodeBatch(const std::vector<std::string>& fens)
{
    if (static_cast<int>(fens.size()) != kSmallBatchSize)
    {
        throw std::runtime_error("TensorRT engine expects batch size " + std::to_string(kSmallBatchSize)
                                 + ", but got " + std::to_string(fens.size()));
    }

    EncodedBatch encoded;
    encoded.features.assign(fens.size() * kSmallBoardSize * kSmallBoardSize * kSmallFeaturePlanes, 0.0F);
    encoded.boards.reserve(fens.size());

    for (size_t idx = 0; idx < fens.size(); ++idx)
    {
        chess::Board board(fens[idx]);
        encoded.boards.push_back(board);

        const auto fillPlane = [&](int plane, float value) {
            for (int r = 0; r < kSmallBoardSize; ++r)
            {
                for (int c = 0; c < kSmallBoardSize; ++c)
                {
                    encoded.features[encodeIndex(static_cast<int>(idx), r, c, plane)] = value;
                }
            }
        };

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
                    encoded.features[encodeIndex(
                        static_cast<int>(idx), kSmallBoardSize - 1 - rank, file, plane)]
                        = 1.0F;
                }
            }
        }

        fillPlane(12, board.sideToMove() == chess::Color::WHITE ? 1.0F : 0.0F);

        const auto enPassant = board.enpassantSq();
        if (enPassant.is_valid())
        {
            const int idxSq = enPassant.index();
            const int rank = idxSq / kSmallBoardSize;
            const int file = idxSq % kSmallBoardSize;
            encoded.features[encodeIndex(
                static_cast<int>(idx), kSmallBoardSize - 1 - rank, file, 13)]
                = 1.0F;
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

        const float fiftyMove = std::min(board.halfMoveClock() / 100.0F, 1.0F);
        fillPlane(18, fiftyMove);
    }

    return encoded;
}

RawOutputs runInference(EngineContext& ctx, const std::vector<float>& input)
{
    auto& engine = *ctx.engine;

    const auto inputDims = engine.getTensorShape(ctx.inputName.c_str());
    const auto policyDims = engine.getTensorShape(ctx.policyOutputName.c_str());
    const auto valueDims = engine.getTensorShape(ctx.valueOutputName.c_str());

    const auto expectedInput = nvinfer1::Dims4{kSmallBatchSize, kSmallBoardSize, kSmallBoardSize, kSmallFeaturePlanes};
    if (volume(inputDims) != volume(expectedInput))
    {
        throw std::runtime_error("Small-model input tensor has unexpected size.");
    }

    const auto expectedPolicy = nvinfer1::Dims2{kSmallBatchSize, kSmallPolicySize};
    if (volume(policyDims) != volume(expectedPolicy))
    {
        throw std::runtime_error("Small-model policy tensor has unexpected size.");
    }

    const auto expectedValue = nvinfer1::Dims2{kSmallBatchSize, kSmallValueSize};
    if (volume(valueDims) != volume(expectedValue))
    {
        throw std::runtime_error("Small-model value tensor has unexpected size.");
    }

    const auto inputType = engine.getTensorDataType(ctx.inputName.c_str());
    const auto policyType = engine.getTensorDataType(ctx.policyOutputName.c_str());
    const auto valueType = engine.getTensorDataType(ctx.valueOutputName.c_str());

    std::unique_ptr<nvinfer1::IExecutionContext> executionContext(engine.createExecutionContext());
    if (!executionContext)
    {
        throw std::runtime_error("Failed to create small-model execution context.");
    }

    if (!executionContext->setInputShape(ctx.inputName.c_str(), expectedInput))
    {
        throw std::runtime_error("Failed to set small-model input shape.");
    }

    DeviceBuffer inputDevice(volume(inputDims) * element_size(inputType));
    DeviceBuffer policyDevice(volume(policyDims) * element_size(policyType));
    DeviceBuffer valueDevice(volume(valueDims) * element_size(valueType));
    CudaStream stream;

    std::vector<__half> inputHalf;
    std::vector<__half> policyHalf;
    std::vector<__half> valueHalf;

    const void* inputHostData = nullptr;
    size_t inputBytes = 0;

    if (inputType == nvinfer1::DataType::kFLOAT)
    {
        inputHostData = input.data();
        inputBytes = input.size() * sizeof(float);
    }
    else if (inputType == nvinfer1::DataType::kHALF)
    {
        inputHalf.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i)
        {
            inputHalf[i] = __float2half(input[i]);
        }
        inputHostData = inputHalf.data();
        inputBytes = inputHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported small-model input tensor type.");
    }

    if (cudaMemcpyAsync(inputDevice.data(), inputHostData, inputBytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (small H2D) failed.");
    }

    if (!executionContext->setTensorAddress(ctx.inputName.c_str(), inputDevice.data())
        || !executionContext->setTensorAddress(ctx.policyOutputName.c_str(), policyDevice.data())
        || !executionContext->setTensorAddress(ctx.valueOutputName.c_str(), valueDevice.data()))
    {
        throw std::runtime_error("Failed to set small-model tensor addresses.");
    }

    if (!executionContext->enqueueV3(stream.get()))
    {
        throw std::runtime_error("Small-model TensorRT enqueue failed.");
    }

    RawOutputs outputs;
    outputs.policy.resize(volume(policyDims));
    outputs.wdl.resize(volume(valueDims));

    void* policyHostData = nullptr;
    size_t policyBytes = 0;

    if (policyType == nvinfer1::DataType::kFLOAT)
    {
        policyHostData = outputs.policy.data();
        policyBytes = outputs.policy.size() * sizeof(float);
    }
    else if (policyType == nvinfer1::DataType::kHALF)
    {
        policyHalf.resize(outputs.policy.size());
        policyHostData = policyHalf.data();
        policyBytes = policyHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported small-model policy tensor type.");
    }

    void* valueHostData = nullptr;
    size_t valueBytes = 0;

    if (valueType == nvinfer1::DataType::kFLOAT)
    {
        valueHostData = outputs.wdl.data();
        valueBytes = outputs.wdl.size() * sizeof(float);
    }
    else if (valueType == nvinfer1::DataType::kHALF)
    {
        valueHalf.resize(outputs.wdl.size());
        valueHostData = valueHalf.data();
        valueBytes = valueHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported small-model value tensor type.");
    }

    if (cudaMemcpyAsync(policyHostData, policyDevice.data(), policyBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (small policy D2H) failed.");
    }

    if (cudaMemcpyAsync(valueHostData, valueDevice.data(), valueBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (small value D2H) failed.");
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaStreamSynchronize (small) failed.");
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

std::vector<std::string> selectMoves(
    const std::vector<float>& logits, const std::vector<chess::Board>& boards, float temperature)
{
    if (boards.size() * kSmallPolicySize != logits.size())
    {
        throw std::runtime_error("Mismatch between small-model logits and batch size.");
    }

    auto& vocab = getPolicyVocab();
    std::vector<std::string> moves;
    moves.reserve(boards.size());

    std::mt19937 rng(std::random_device{}());

    std::vector<float> row(kSmallPolicySize);
    std::vector<char> legalMask(kSmallPolicySize);
    std::vector<float> scaled(kSmallPolicySize);
    std::vector<double> probs(kSmallPolicySize);

    for (size_t idx = 0; idx < boards.size(); ++idx)
    {
        const float* rowStart = logits.data() + static_cast<ptrdiff_t>(idx * kSmallPolicySize);
        std::copy(rowStart, rowStart + kSmallPolicySize, row.begin());
        std::fill(legalMask.begin(), legalMask.end(), 0);

        chess::Movelist legalMoves;
        chess::movegen::legalmoves(legalMoves, boards[idx]);

        std::vector<int> legalIndices;
        legalIndices.reserve(legalMoves.size());

        for (const auto& move : legalMoves)
        {
            std::string uci = chess::uci::moveToUci(move, boards[idx].chess960());
            if (!uci.empty() && uci.back() == 'n')
            {
                uci.pop_back();
            }

            if (auto mapped = vocab.indexOf(uci))
            {
                if (!legalMask[*mapped])
                {
                    legalMask[*mapped] = 1;
                    legalIndices.push_back(*mapped);
                }
            }
        }

        if (!legalIndices.empty())
        {
            for (int i = 0; i < kSmallPolicySize; ++i)
            {
                if (!legalMask[i])
                {
                    row[i] = kIllegalScore;
                }
            }
        }

        int selectedIndex = -1;

        if (temperature <= 0.0F || legalIndices.size() <= 1)
        {
            selectedIndex = static_cast<int>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
        }
        else
        {
            float maxVal = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < kSmallPolicySize; ++i)
            {
                const float val = row[i] / temperature;
                scaled[i] = val;
                if (val > maxVal)
                {
                    maxVal = val;
                }
            }

            double sum = 0.0;
            for (int i = 0; i < kSmallPolicySize; ++i)
            {
                const double expVal = std::exp(static_cast<double>(scaled[i] - maxVal));
                probs[i] = expVal;
                sum += expVal;
            }

            if (sum <= 0.0)
            {
                selectedIndex = static_cast<int>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
            }
            else
            {
                for (double& p : probs)
                {
                    p /= sum;
                }
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                selectedIndex = dist(rng);
            }
        }

        if (selectedIndex < 0 || selectedIndex >= kSmallPolicySize)
        {
            throw std::runtime_error("Failed to select small-model move.");
        }

        moves.push_back(vocab.at(selectedIndex));
    }

    return moves;
}

SmallInferenceResult runOnce(const std::vector<std::string>& fens, float temperature, EngineContext& ctx)
{
    using clock = std::chrono::high_resolution_clock;
    const auto totalStart = clock::now();

    const auto encodeStart = clock::now();
    EncodedBatch encoded = encodeBatch(fens);
    const auto encodeEnd = clock::now();

    const auto inferStart = clock::now();
    RawOutputs rawOutputs = runInference(ctx, encoded.features);
    const auto inferEnd = clock::now();

    const auto sampleStart = clock::now();
    std::vector<std::string> moves = selectMoves(rawOutputs.policy, encoded.boards, temperature);
    const auto sampleEnd = clock::now();

    if (rawOutputs.wdl.size() != encoded.boards.size() * static_cast<size_t>(kSmallValueSize))
    {
        throw std::runtime_error("Mismatch between small-model value outputs and batch size.");
    }

    std::vector<std::array<float, kSmallValueSize>> wdlVectors;
    wdlVectors.reserve(encoded.boards.size());
    for (size_t idx = 0; idx < encoded.boards.size(); ++idx)
    {
        std::array<float, kSmallValueSize> entry{};
        for (int j = 0; j < kSmallValueSize; ++j)
        {
            entry[static_cast<size_t>(j)]
                = rawOutputs.wdl[idx * static_cast<size_t>(kSmallValueSize) + static_cast<size_t>(j)];
        }
        wdlVectors.push_back(entry);
    }

    const auto totalEnd = clock::now();

    auto durationMs = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    InferenceTimings timings{
        .encodeMs = durationMs(encodeStart, encodeEnd),
        .inferMs = durationMs(inferStart, inferEnd),
        .sampleMs = durationMs(sampleStart, sampleEnd),
        .totalMs = durationMs(totalStart, totalEnd),
    };

    return SmallInferenceResult{std::move(moves), std::move(wdlVectors), timings};
}

} // namespace small_impl

} // namespace

SmallInferenceResult small_generate_moves(
    const std::vector<std::string>& fens,
    float temperature,
    int iterations,
    const std::string& engine_path)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument("iterations must be >= 1");
    }

    auto& ctx = small_impl::getEngineContext(engine_path);

    InferenceTimings cumulative{};
    SmallInferenceResult lastResult;

    for (int i = 0; i < iterations; ++i)
    {
        SmallInferenceResult result = small_impl::runOnce(fens, temperature, ctx);
        cumulative.encodeMs += result.timings.encodeMs;
        cumulative.inferMs += result.timings.inferMs;
        cumulative.sampleMs += result.timings.sampleMs;
        cumulative.totalMs += result.timings.totalMs;

        if (i == iterations - 1)
        {
            lastResult = std::move(result);
        }
    }

    lastResult.timings.encodeMs = cumulative.encodeMs / static_cast<double>(iterations);
    lastResult.timings.inferMs = cumulative.inferMs / static_cast<double>(iterations);
    lastResult.timings.sampleMs = cumulative.sampleMs / static_cast<double>(iterations);
    lastResult.timings.totalMs = cumulative.totalMs / static_cast<double>(iterations);

    return lastResult;
}

size_t small_policy_size()
{
    return kSmallPolicySize;
}

const std::vector<std::string>& small_policy_vocabulary()
{
    return small_impl::getPolicyVocab().tokensRef();
}

} // namespace chessrl
