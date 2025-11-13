#include "mcts/single_inference.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
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

constexpr int kLeelaBoardSize = 8;
constexpr int kLeelaFeaturePlanes = 112;
constexpr int kLeelaPolicySize = 1858;
constexpr int kLeelaValueSize = 3;
constexpr int kLeelaHistoryLength = 8;

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
                const size_t total = volume(dims);
                if (total == static_cast<size_t>(kSmallPolicySize) && mPolicyOutputName.empty())
                {
                    mPolicyOutputName = name;
                }
                else if (total == static_cast<size_t>(kSmallValueSize) && mValueOutputName.empty())
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

    std::string resolved = resolveEnginePath(engine_path, "model_single.trt");
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

    std::vector<float> features = encodeSmallFeatures(board);
    SmallEngineContext& ctx = getSmallEngineContext(engine_path);
    SmallRawOutputs raw = runSmallInference(ctx, features);

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
    if (board.sideToMove() == chess::Color::WHITE)
    {
        std::swap(result.wdl[0], result.wdl[2]);
    }
    return result;
}

// -------------------------------------------------------------
// Leela-model single-batch inference
// -------------------------------------------------------------

struct LeelaRawOutputs
{
    std::vector<float> policy;
    std::vector<float> valueWinner;
    std::vector<float> valueQ;
};

class LeelaEngineContext
{
public:
    explicit LeelaEngineContext(std::string enginePath)
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
                const size_t total = volume(dims);
                if (total == static_cast<size_t>(kLeelaPolicySize) && mPolicyOutputName.empty())
                {
                    mPolicyOutputName = name;
                }
                else if (total == static_cast<size_t>(kLeelaValueSize) && mValueWinnerOutputName.empty())
                {
                    mValueWinnerOutputName = name;
                }
                else if (total == static_cast<size_t>(kLeelaValueSize) && mValueQOutputName.empty())
                {
                    mValueQOutputName = name;
                }
            }
        }

        if (mInputName.empty() || mPolicyOutputName.empty() || mValueWinnerOutputName.empty() || mValueQOutputName.empty())
        {
            throw std::runtime_error("Failed to determine Leela-model tensor names.");
        }
    }

    nvinfer1::ICudaEngine& engine() { return *mEngine; }
    const std::string& inputName() const { return mInputName; }
    const std::string& policyName() const { return mPolicyOutputName; }
    const std::string& valueWinnerName() const { return mValueWinnerOutputName; }
    const std::string& valueQName() const { return mValueQOutputName; }

private:
    std::string mEnginePath;
    Logger mLogger;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::string mInputName;
    std::string mPolicyOutputName;
    std::string mValueWinnerOutputName;
    std::string mValueQOutputName;
};

LeelaEngineContext& getLeelaEngineContext(const std::string& engine_path)
{
    static auto* mutex = new std::mutex();
    static auto* contexts = new std::unordered_map<std::string, std::unique_ptr<LeelaEngineContext>>();

    std::string resolved = resolveEnginePath(engine_path, "leela_single.trt");
    std::lock_guard<std::mutex> lock(*mutex);
    auto it = contexts->find(resolved);
    if (it == contexts->end())
    {
        auto ctx = std::make_unique<LeelaEngineContext>(resolved);
        auto [inserted, _] = contexts->emplace(resolved, std::move(ctx));
        return *inserted->second;
    }
    return *it->second;
}

std::vector<std::string> splitByChar(const std::string& input, char delimiter)
{
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string part;
    while (std::getline(ss, part, delimiter))
    {
        result.push_back(part);
    }
    return result;
}

std::vector<std::string> splitFen(const std::string& fen)
{
    std::vector<std::string> parts;
    std::istringstream ss(fen);
    std::string part;
    while (ss >> part)
    {
        parts.push_back(part);
    }
    return parts;
}

std::string reorderCastling(const std::string& rights)
{
    if (rights.empty() || rights == "-")
    {
        return "-";
    }

    std::string ordered;
    auto appendIfPresent = [&](char ch) {
        if (rights.find(ch) != std::string::npos)
        {
            ordered.push_back(ch);
        }
    };

    appendIfPresent('K');
    appendIfPresent('Q');
    appendIfPresent('k');
    appendIfPresent('q');

    return ordered.empty() ? "-" : ordered;
}

std::string mirrorFen(const chess::Board& board)
{
    const std::string fen = board.getFen();
    const auto fields = splitFen(fen);
    if (fields.size() != 6)
    {
        return fen;
    }

    std::stringstream placementStream;
    const auto ranks = splitByChar(fields[0], '/');
    for (size_t i = 0; i < ranks.size(); ++i)
    {
        std::string rank = ranks[ranks.size() - 1 - i];
        for (char& c : rank)
        {
            if (std::islower(static_cast<unsigned char>(c)))
            {
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            }
            else if (std::isupper(static_cast<unsigned char>(c)))
            {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
        }
        if (i != 0)
        {
            placementStream << '/';
        }
        placementStream << rank;
    }
    const std::string placement = placementStream.str();

    std::string stm = fields[1];
    if (stm == "w")
    {
        stm = "b";
    }
    else if (stm == "b")
    {
        stm = "w";
    }
    else
    {
        throw std::runtime_error("Invalid active color in FEN while mirroring board.");
    }

    std::string castling = fields[2];
    if (castling != "-")
    {
        std::string converted;
        converted.reserve(castling.size());
        for (char c : castling)
        {
            if (std::isalpha(static_cast<unsigned char>(c)))
            {
                if (std::islower(static_cast<unsigned char>(c)))
                {
                    converted.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
                }
                else
                {
                    converted.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                }
            }
        }
        castling = reorderCastling(converted);
    }

    std::string ep = fields[3];
    if (ep != "-")
    {
        if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && ep[1] >= '1' && ep[1] <= '8')
        {
            const int rank = ep[1] - '0';
            const int mirroredRank = 9 - rank;
            ep[1] = static_cast<char>('0' + mirroredRank);
        }
        else
        {
            ep = "-";
        }
    }

    const std::string halfmove = fields.size() > 4 ? fields[4] : "0";
    const std::string fullmove = fields.size() > 5 ? fields[5] : "1";

    std::ostringstream os;
    os << placement << ' ' << stm << ' ' << castling << ' ' << ep << ' ' << halfmove << ' ' << fullmove;
    return os.str();
}

chess::Board mirrorBoard(const chess::Board& board)
{
    return chess::Board(mirrorFen(board));
}

std::string mirrorUciRanks(const std::string& move)
{
    if (move.size() < 4)
    {
        return move;
    }
    auto mirrorRankChar = [](char c) -> char {
        if (c < '1' || c > '8')
        {
            return c;
        }
        const int rank = c - '0';
        const int mirrored = 9 - rank;
        return static_cast<char>('0' + mirrored);
    };
    std::string mirrored = move;
    mirrored[1] = mirrorRankChar(mirrored[1]);
    mirrored[3] = mirrorRankChar(mirrored[3]);
    return mirrored;
}

std::string adjustCastling(const std::string& move, const std::string& rights)
{
    if (move == "e1h1" && rights.find('K') != std::string::npos)
    {
        return "e1g1";
    }
    if (move == "e1a1" && rights.find('Q') != std::string::npos)
    {
        return "e1c1";
    }
    if (move == "e8h8" && rights.find('k') != std::string::npos)
    {
        return "e8g8";
    }
    if (move == "e8a8" && rights.find('q') != std::string::npos)
    {
        return "e8c8";
    }
    return move;
}

std::string extractCastlingRights(const chess::Board& board)
{
    const auto fields = splitFen(board.getFen());
    if (fields.size() >= 3)
    {
        return fields[2];
    }
    return "-";
}

size_t encodeLeelaIndex(int plane, int row, int col)
{
    return ((static_cast<size_t>(plane) * kLeelaBoardSize + static_cast<size_t>(row)) * kLeelaBoardSize)
        + static_cast<size_t>(col);
}

void writeConstantPlane(std::vector<float>& features, int planeIdx, float value)
{
    if (value == 0.0F)
    {
        return;
    }
    for (int row = 0; row < kLeelaBoardSize; ++row)
    {
        for (int col = 0; col < kLeelaBoardSize; ++col)
        {
            features[encodeLeelaIndex(planeIdx, row, col)] = value;
        }
    }
}

void writePiecePlanes(const chess::Board& board, int planeOffset, std::vector<float>& features)
{
    const std::array<chess::Color, 2> colors{chess::Color::WHITE, chess::Color::BLACK};
    const std::array<chess::PieceType, 6> pieceTypes{
        chess::PieceType::PAWN,
        chess::PieceType::KNIGHT,
        chess::PieceType::BISHOP,
        chess::PieceType::ROOK,
        chess::PieceType::QUEEN,
        chess::PieceType::KING};

    int plane = planeOffset;
    for (const auto color : colors)
    {
        for (const auto pieceType : pieceTypes)
        {
            chess::Bitboard bb = board.pieces(pieceType, color);
            while (bb)
            {
                const int squareIndex = static_cast<int>(bb.pop());
                const int row = squareIndex / kLeelaBoardSize;
                const int col = squareIndex % kLeelaBoardSize;
                features[encodeLeelaIndex(plane, row, col)] = 1.0F;
            }
            ++plane;
        }
    }
}

struct LeelaEncoded
{
    std::vector<float> features;
    chess::Board maskBoard;
    bool isBlackToMove{false};
    std::string castlingRights;
};

LeelaEncoded encodeLeelaInput(std::string_view originFen, const std::vector<std::string>& moveHistory)
{
    const std::string fenToUse = originFen.empty() ? std::string(chess::constants::STARTPOS) : std::string(originFen);
    chess::Board board(fenToUse);

    std::vector<chess::Board> states;
    states.reserve(moveHistory.size() + 1);
    states.push_back(board);

    for (const auto& uci : moveHistory)
    {
        const chess::Move move = chess::uci::uciToMove(board, uci);
        if (move == chess::Move::NO_MOVE)
        {
            throw std::runtime_error("Failed to parse UCI move: " + uci);
        }
        board.makeMove(move);
        states.push_back(board);
    }

    const chess::Board& finalBoard = states.back();

    std::array<std::optional<chess::Board>, kLeelaHistoryLength> history{};
    for (int h = 0; h < kLeelaHistoryLength; ++h)
    {
        if (states.size() > static_cast<size_t>(h))
        {
            history[h] = states[states.size() - 1 - static_cast<size_t>(h)];
        }
    }

    LeelaEncoded encoded;
    encoded.features.assign(kLeelaFeaturePlanes * kLeelaBoardSize * kLeelaBoardSize, 0.0F);

    encoded.isBlackToMove = finalBoard.sideToMove() == chess::Color::BLACK;
    encoded.castlingRights = extractCastlingRights(finalBoard);

    auto featureHistory = history;
    if (encoded.isBlackToMove)
    {
        for (auto& snapshot : featureHistory)
        {
            if (snapshot)
            {
                snapshot = mirrorBoard(*snapshot);
            }
        }
        encoded.maskBoard = mirrorBoard(finalBoard);
    }
    else
    {
        encoded.maskBoard = finalBoard;
    }

    for (int h = 0; h < kLeelaHistoryLength; ++h)
    {
        if (featureHistory[h])
        {
            const int planeOffset = h * 13;
            writePiecePlanes(*featureHistory[h], planeOffset, encoded.features);
        }
    }

    const auto& currentSnapshot = featureHistory[0];
    if (!currentSnapshot)
    {
        throw std::runtime_error("Leela encoding expected current snapshot to be available.");
    }

    const auto castling = currentSnapshot->castlingRights();
    writeConstantPlane(
        encoded.features,
        104,
        castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);
    writeConstantPlane(
        encoded.features,
        105,
        castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
    writeConstantPlane(
        encoded.features,
        106,
        castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);
    writeConstantPlane(
        encoded.features,
        107,
        castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);

    if (encoded.isBlackToMove)
    {
        writeConstantPlane(encoded.features, 108, 1.0F);
    }
    writeConstantPlane(encoded.features, 111, 1.0F);

    return encoded;
}

LeelaRawOutputs runLeelaInference(LeelaEngineContext& ctx, const std::vector<float>& input)
{
    const size_t kInputElements
        = static_cast<size_t>(kLeelaFeaturePlanes) * kLeelaBoardSize * kLeelaBoardSize;
    const size_t kPolicyElements = static_cast<size_t>(kLeelaPolicySize);
    const size_t kValueElements = static_cast<size_t>(kLeelaValueSize);

    if (input.size() != kInputElements)
    {
        throw std::runtime_error("Leela-model single-batch received unexpected feature vector size.");
    }

    nvinfer1::ICudaEngine& engine = ctx.engine();

    std::unique_ptr<nvinfer1::IExecutionContext> exec(engine.createExecutionContext());
    if (!exec)
    {
        throw std::runtime_error("Failed to create Leela-model execution context.");
    }

    const auto requestedInput = nvinfer1::Dims4{1, kLeelaFeaturePlanes, kLeelaBoardSize, kLeelaBoardSize};
    if (!exec->setInputShape(ctx.inputName().c_str(), requestedInput))
    {
        throw std::runtime_error("Failed to set Leela-model input shape.");
    }

    const auto inputType = engine.getTensorDataType(ctx.inputName().c_str());
    const auto policyType = engine.getTensorDataType(ctx.policyName().c_str());
    const auto winnerType = engine.getTensorDataType(ctx.valueWinnerName().c_str());
    const auto qType = engine.getTensorDataType(ctx.valueQName().c_str());

    DeviceBuffer inputDevice(kInputElements * getElementSize(inputType));
    DeviceBuffer policyDevice(kPolicyElements * getElementSize(policyType));
    DeviceBuffer winnerDevice(kValueElements * getElementSize(winnerType));
    DeviceBuffer qDevice(kValueElements * getElementSize(qType));
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
        throw std::runtime_error("Unsupported Leela-model input tensor type.");
    }

    if (cudaMemcpyAsync(inputDevice.data(), inputHost, inputBytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (Leela single H2D) failed.");
    }

    if (!exec->setTensorAddress(ctx.inputName().c_str(), inputDevice.data())
        || !exec->setTensorAddress(ctx.policyName().c_str(), policyDevice.data())
        || !exec->setTensorAddress(ctx.valueWinnerName().c_str(), winnerDevice.data())
        || !exec->setTensorAddress(ctx.valueQName().c_str(), qDevice.data()))
    {
        throw std::runtime_error("Failed to set Leela-model tensor addresses.");
    }

    if (!exec->enqueueV3(stream.get()))
    {
        throw std::runtime_error("Leela-model single-batch TensorRT enqueue failed.");
    }

    LeelaRawOutputs outputs;
    outputs.policy.resize(kPolicyElements);
    outputs.valueWinner.resize(kValueElements);
    outputs.valueQ.resize(kValueElements);

    std::vector<__half> policyHalf;
    std::vector<__half> winnerHalf;
    std::vector<__half> qHalf;

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
        throw std::runtime_error("Unsupported Leela-model policy tensor type.");
    }

    void* winnerHost = nullptr;
    size_t winnerBytes = 0;

    if (winnerType == nvinfer1::DataType::kFLOAT)
    {
        winnerHost = outputs.valueWinner.data();
        winnerBytes = outputs.valueWinner.size() * sizeof(float);
    }
    else if (winnerType == nvinfer1::DataType::kHALF)
    {
        winnerHalf.resize(outputs.valueWinner.size());
        winnerHost = winnerHalf.data();
        winnerBytes = winnerHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported Leela-model value_winner tensor type.");
    }

    void* qHost = nullptr;
    size_t qBytes = 0;

    if (qType == nvinfer1::DataType::kFLOAT)
    {
        qHost = outputs.valueQ.data();
        qBytes = outputs.valueQ.size() * sizeof(float);
    }
    else if (qType == nvinfer1::DataType::kHALF)
    {
        qHalf.resize(outputs.valueQ.size());
        qHost = qHalf.data();
        qBytes = qHalf.size() * sizeof(__half);
    }
    else
    {
        throw std::runtime_error("Unsupported Leela-model value_q tensor type.");
    }

    if (cudaMemcpyAsync(policyHost, policyDevice.data(), policyBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (Leela single policy D2H) failed.");
    }
    if (cudaMemcpyAsync(winnerHost, winnerDevice.data(), winnerBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (Leela single value_winner D2H) failed.");
    }
    if (cudaMemcpyAsync(qHost, qDevice.data(), qBytes, cudaMemcpyDeviceToHost, stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync (Leela single value_q D2H) failed.");
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        throw std::runtime_error("cudaStreamSynchronize (Leela single) failed.");
    }

    if (policyType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.policy.size(); ++i)
        {
            outputs.policy[i] = __half2float(policyHalf[i]);
        }
    }
    if (winnerType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.valueWinner.size(); ++i)
        {
            outputs.valueWinner[i] = __half2float(winnerHalf[i]);
        }
    }
    if (qType == nvinfer1::DataType::kHALF)
    {
        for (size_t i = 0; i < outputs.valueQ.size(); ++i)
        {
            outputs.valueQ[i] = __half2float(qHalf[i]);
        }
    }

    return outputs;
}

LeelaPolicyValue evaluateLeela(
    std::string_view origin_fen,
    const std::vector<std::string>& move_history,
    const std::string& engine_path)
{
    LeelaEncoded encoded = encodeLeelaInput(origin_fen, move_history);
    LeelaEngineContext& ctx = getLeelaEngineContext(engine_path);
    LeelaRawOutputs raw = runLeelaInference(ctx, encoded.features);

    if (raw.policy.size() != static_cast<size_t>(kLeelaPolicySize)
        || raw.valueWinner.size() != static_cast<size_t>(kLeelaValueSize)
        || raw.valueQ.size() != static_cast<size_t>(kLeelaValueSize))
    {
        throw std::runtime_error("Unexpected output sizes from Leela single-batch inference.");
    }

    LeelaPolicyValue result;
    result.policy_logits = std::move(raw.policy);
    for (int i = 0; i < kLeelaValueSize; ++i)
    {
        result.value_winner_logits[static_cast<size_t>(i)] = raw.valueWinner[static_cast<size_t>(i)];
        result.value_q_logits[static_cast<size_t>(i)] = raw.valueQ[static_cast<size_t>(i)];
    }
    result.is_black_to_move = encoded.isBlackToMove;
    result.castling_rights = std::move(encoded.castlingRights);
    return result;
}

} // namespace

SmallPolicyValue evaluate_small_position(std::string_view fen, const std::string& engine_path)
{
    return evaluateSmall(fen, engine_path);
}

LeelaPolicyValue evaluate_leela_position(
    std::string_view origin_fen,
    const std::vector<std::string>& move_history,
    const std::string& engine_path)
{
    return evaluateLeela(origin_fen, move_history, engine_path);
}

} // namespace chessrl::mcts::single

