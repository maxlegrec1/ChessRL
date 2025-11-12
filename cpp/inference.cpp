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
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chess-library/include/chess.hpp"
#include "cuda_fp16.h"

namespace chessrl
{

namespace
{

constexpr int kBoardSize = 8;
constexpr int kFeaturePlanes = 19;
constexpr int kPolicySize = 1929;
constexpr int kBatchSize = 1024;
constexpr int kValueSize = 3;

struct RawInferenceOutputs
{
    std::vector<float> policy;
    std::vector<float> wdl;
};

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
    return std::accumulate(dims.d, dims.d + dims.nbDims, static_cast<size_t>(1),
        [](size_t a, int64_t b) { return a * static_cast<size_t>(b); });
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

struct EngineContext
{
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
            if (total == static_cast<size_t>(kBatchSize) * kPolicySize && policyOutputName.empty())
            {
                policyOutputName = name;
            }
            else if (total == static_cast<size_t>(kBatchSize) * kValueSize && valueOutputName.empty())
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
            throw std::runtime_error("Failed to determine input/output tensor names.");
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

    std::string resolvedPath = engine_path.empty() ? std::string("model.trt") : engine_path;
    try
    {
        resolvedPath = std::filesystem::absolute(resolvedPath).string();
    }
    catch (...)
    {
        // fall back to provided path if canonicalization fails
    }

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

struct PolicyVocab
{
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

        if (tokens.size() != kPolicySize)
        {
            throw std::runtime_error("Unexpected policy vocab size: " + std::to_string(tokens.size()));
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
    return (((static_cast<size_t>(batch) * kBoardSize + static_cast<size_t>(row)) * kBoardSize
             + static_cast<size_t>(col))
            * kFeaturePlanes)
        + static_cast<size_t>(plane);
}

struct EncodedBatch
{
    std::vector<float> features;
    std::vector<chess::Board> boards;
};

EncodedBatch encodeBatch(const std::vector<std::string>& fens)
{
    if (static_cast<int>(fens.size()) != kBatchSize)
    {
        throw std::runtime_error("TensorRT engine expects batch size " + std::to_string(kBatchSize)
                                 + ", but got " + std::to_string(fens.size()));
    }

    EncodedBatch encoded;
    encoded.features.assign(fens.size() * kBoardSize * kBoardSize * kFeaturePlanes, 0.0F);
    encoded.boards.reserve(fens.size());

    for (size_t idx = 0; idx < fens.size(); ++idx)
    {
        chess::Board board(fens[idx]);
        encoded.boards.push_back(board);

        const auto fillPlane = [&](int plane, float value) {
            for (int r = 0; r < kBoardSize; ++r)
            {
                for (int c = 0; c < kBoardSize; ++c)
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
                    const int rank = squareIndex / kBoardSize;
                    const int file = squareIndex % kBoardSize;
                    const int plane = base + offset;
                    encoded.features[encodeIndex(static_cast<int>(idx), kBoardSize - 1 - rank, file, plane)] = 1.0F;
                }
            }
        }

        fillPlane(12, board.sideToMove() == chess::Color::WHITE ? 1.0F : 0.0F);

        const auto enPassant = board.enpassantSq();
        if (enPassant.is_valid())
        {
            const int idxSq = enPassant.index();
            const int rank = idxSq / kBoardSize;
            const int file = idxSq % kBoardSize;
            encoded.features[encodeIndex(static_cast<int>(idx), kBoardSize - 1 - rank, file, 13)] = 1.0F;
        }

        const auto castling = board.castlingRights();
        fillPlane(14, castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
        fillPlane(15, castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);
        fillPlane(16, castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
        fillPlane(17, castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);

        const float fiftyMove = std::min(board.halfMoveClock() / 100.0F, 1.0F);
        fillPlane(18, fiftyMove);
    }

    return encoded;
}

RawInferenceOutputs runInference(EngineContext& ctx, const std::vector<float>& input)
{
    auto& engine = *ctx.engine;

    const auto inputDims = engine.getTensorShape(ctx.inputName.c_str());
    const auto policyDims = engine.getTensorShape(ctx.policyOutputName.c_str());
    const auto valueDims = engine.getTensorShape(ctx.valueOutputName.c_str());

    const auto expectedInput = nvinfer1::Dims4{kBatchSize, kBoardSize, kBoardSize, kFeaturePlanes};
    if (volume(inputDims) != volume(expectedInput))
    {
        throw std::runtime_error("Input tensor has unexpected size.");
    }

    const auto expectedPolicy = nvinfer1::Dims2{kBatchSize, kPolicySize};
    if (volume(policyDims) != volume(expectedPolicy))
    {
        throw std::runtime_error("Policy tensor has unexpected size.");
    }

    const auto expectedValue = nvinfer1::Dims2{kBatchSize, kValueSize};
    if (volume(valueDims) != volume(expectedValue))
    {
        throw std::runtime_error("Value tensor has unexpected size.");
    }

    const auto inputType = engine.getTensorDataType(ctx.inputName.c_str());
    const auto policyType = engine.getTensorDataType(ctx.policyOutputName.c_str());
    const auto valueType = engine.getTensorDataType(ctx.valueOutputName.c_str());

    std::unique_ptr<nvinfer1::IExecutionContext> executionContext(engine.createExecutionContext());
    if (!executionContext)
    {
        throw std::runtime_error("Failed to create execution context.");
    }

    if (!executionContext->setInputShape(ctx.inputName.c_str(), expectedInput))
    {
        throw std::runtime_error("Failed to set input shape.");
    }

    DeviceBuffer inputDevice(volume(inputDims) * getElementSize(inputType));
    DeviceBuffer policyDevice(volume(policyDims) * getElementSize(policyType));
    DeviceBuffer valueDevice(volume(valueDims) * getElementSize(valueType));
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
        throw std::runtime_error("Unsupported input tensor type.");
    }

    if (cudaMemcpyAsync(inputDevice.data(), inputHostData, inputBytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (H2D) failed.");
    }

    if (!executionContext->setTensorAddress(ctx.inputName.c_str(), inputDevice.data())
        || !executionContext->setTensorAddress(ctx.policyOutputName.c_str(), policyDevice.data())
        || !executionContext->setTensorAddress(ctx.valueOutputName.c_str(), valueDevice.data()))
    {
        throw std::runtime_error("Failed to set tensor addresses.");
    }

    if (!executionContext->enqueueV3(stream.get()))
    {
        throw std::runtime_error("TensorRT enqueue failed.");
    }

    RawInferenceOutputs outputs;
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
        throw std::runtime_error("Unsupported policy tensor type.");
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
        throw std::runtime_error("Unsupported value tensor type.");
    }

    if (cudaMemcpyAsync(policyHostData, policyDevice.data(), policyBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (policy D2H) failed.");
    }

    if (cudaMemcpyAsync(valueHostData, valueDevice.data(), valueBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (value D2H) failed.");
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaStreamSynchronize failed.");
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
    if (boards.size() * kPolicySize != logits.size())
    {
        throw std::runtime_error("Mismatch between logits and batch size.");
    }

    auto& vocab = getPolicyVocab();
    std::vector<std::string> moves;
    moves.reserve(boards.size());

    std::mt19937 rng(std::random_device{}());

    constexpr float kIllegalScore = -1e9F;

    std::vector<float> row(kPolicySize);
    std::vector<char> legalMask(kPolicySize);

    for (size_t idx = 0; idx < boards.size(); ++idx)
    {
        const float* rowStart = logits.data() + static_cast<ptrdiff_t>(idx * kPolicySize);
        std::copy(rowStart, rowStart + kPolicySize, row.begin());
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
                legalMask[*mapped] = 1;
                legalIndices.push_back(*mapped);
            }
        }

        if (!legalIndices.empty())
        {
            for (int i = 0; i < kPolicySize; ++i)
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
            std::vector<float> scaled(kPolicySize);
            scaled.reserve(kPolicySize);
            float maxVal = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < kPolicySize; ++i)
            {
                const float val = row[i] / temperature;
                scaled[i] = val;
                if (val > maxVal)
                {
                    maxVal = val;
                }
            }

            std::vector<double> probs(kPolicySize);
            double sum = 0.0;
            for (int i = 0; i < kPolicySize; ++i)
            {
                const float val = scaled[i] - maxVal;
                const double expVal = std::exp(static_cast<double>(val));
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

        if (selectedIndex < 0 || selectedIndex >= kPolicySize)
        {
            throw std::runtime_error("Failed to select move.");
        }

        moves.push_back(vocab.at(selectedIndex));
    }

    return moves;
}

MoveGenerationResult runOnce(const std::vector<std::string>& fens, float temperature, EngineContext& ctx)
{
    using clock = std::chrono::high_resolution_clock;
    const auto totalStart = clock::now();

    const auto encodeStart = clock::now();
    EncodedBatch encoded = encodeBatch(fens);
    const auto encodeEnd = clock::now();

    const auto inferStart = clock::now();
    RawInferenceOutputs rawOutputs = runInference(ctx, encoded.features);
    const auto inferEnd = clock::now();

    const auto sampleStart = clock::now();
    std::vector<std::string> moves = selectMoves(rawOutputs.policy, encoded.boards, temperature);
    const auto sampleEnd = clock::now();

    if (rawOutputs.wdl.size() != encoded.boards.size() * static_cast<size_t>(kValueSize))
    {
        throw std::runtime_error("Mismatch between value outputs and batch size.");
    }

    std::vector<std::array<float, kValueSize>> wdlVectors;
    wdlVectors.reserve(encoded.boards.size());
    for (size_t idx = 0; idx < encoded.boards.size(); ++idx)
    {
        std::array<float, kValueSize> entry{};
        for (int j = 0; j < kValueSize; ++j)
        {
            entry[static_cast<size_t>(j)]
                = rawOutputs.wdl[idx * static_cast<size_t>(kValueSize) + static_cast<size_t>(j)];
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

    return MoveGenerationResult{std::move(moves), std::move(wdlVectors), timings};
}

} // namespace

MoveGenerationResult generate_moves(
    const std::vector<std::string>& fens, float temperature, int iterations, const std::string& engine_path)
{
    if (iterations <= 0)
    {
        throw std::invalid_argument("iterations must be >= 1");
    }

    EngineContext& ctx = getEngineContext(engine_path);

    InferenceTimings cumulative{};
    MoveGenerationResult lastResult;

    for (int i = 0; i < iterations; ++i)
    {
        MoveGenerationResult result = runOnce(fens, temperature, ctx);
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

} // namespace chessrl


