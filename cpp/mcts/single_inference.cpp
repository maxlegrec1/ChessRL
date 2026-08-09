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
#include <unordered_set>
#include <utility>
#include <vector>

#include "cuda_fp16.h"

#if defined(__F16C__)
#include <immintrin.h>
#endif

#include "chess-library/include/chess.hpp"
#include "include/trt_utils.hpp"
#include "mcts/expand_kernel.hpp"

namespace chessrl::mcts::single
{
namespace
{

// Process-wide encoder time accumulators. Updated under no lock — relaxed
// atomic increments. Used only for benchmarking and never read by the hot
// path, so contention is one inc per leaf and negligible vs the encoder's
// own bit-unpack cost.
std::atomic<std::uint64_t> g_encoder_calls{0};
std::atomic<std::uint64_t> g_encoder_nanos{0};

// CUDA-graph dispatch counters (benchmark only). graph = batch served by a
// captured+replayed graph; fallback = batch served by the enqueueV3 path.
std::atomic<std::uint64_t> g_graph_batches{0};
std::atomic<std::uint64_t> g_fallback_batches{0};

} // namespace

EncoderTotals encoder_totals()
{
    EncoderTotals t;
    t.calls = g_encoder_calls.load(std::memory_order_relaxed);
    t.nanos = g_encoder_nanos.load(std::memory_order_relaxed);
    return t;
}

void encoder_totals_reset()
{
    g_encoder_calls.store(0, std::memory_order_relaxed);
    g_encoder_nanos.store(0, std::memory_order_relaxed);
}

namespace
{

using chessrl::trt::Logger;
using chessrl::trt::element_size;
using chessrl::trt::volume;
using chessrl::trt::resolve_engine_path;

constexpr int kSmallBoardSize = 8;
// AlphaZero-style stacked input: 8 boards × (12 piece planes + 1 repetition
// plane) + 7 globals (STM, EP, 4× castling, halfmove clock). See
// src/models/small/fen_encoder.py for the matching Python layout — these
// two must agree bit-for-bit (tests/test_encoder_parity.py is gating).
constexpr int kSmallPiecePlanes = 12;
constexpr int kSmallHistoryDepth = 8;
constexpr int kSmallHistBlock = kSmallPiecePlanes + 1; // piece planes + rep
constexpr int kSmallGlobalPlanes = 7;                   // STM, EP, 4 castling, hmc
constexpr int kSmallFeaturePlanes =
    kSmallHistoryDepth * kSmallHistBlock + kSmallGlobalPlanes; // 111
// Plane indices into the global section (same as Python).
constexpr int kSmallPlaneSTM = kSmallHistoryDepth * kSmallHistBlock;       // 104
constexpr int kSmallPlaneEP = kSmallPlaneSTM + 1;                          // 105
constexpr int kSmallPlaneCastlingWK = kSmallPlaneSTM + 2;                  // 106
constexpr int kSmallPlaneCastlingWQ = kSmallPlaneSTM + 3;                  // 107
constexpr int kSmallPlaneCastlingBK = kSmallPlaneSTM + 4;                  // 108
constexpr int kSmallPlaneCastlingBQ = kSmallPlaneSTM + 5;                  // 109
constexpr int kSmallPlaneHMC = kSmallPlaneSTM + 6;                         // 110
constexpr int kSmallPolicySize = 1929;
constexpr int kSmallValueSize = 3;

// Drift guard: the kernel-shared constants in packed_position.hpp MUST
// match the encoder's. If anyone edits one set, this fails to compile.
static_assert(kSmallBoardSize == kPackBoard, "board size drift");
static_assert(kSmallPiecePlanes == kPackPiecePlanes, "piece planes drift");
static_assert(kSmallHistoryDepth == kPackHistory, "history depth drift");
static_assert(kSmallHistBlock == kPackHistBlock, "hist block drift");
static_assert(kSmallFeaturePlanes == kPackPlanes, "plane count drift");
static_assert(kSmallPlaneSTM == kPackPlaneSTM, "STM plane drift");
static_assert(kSmallPlaneEP == kPackPlaneEP, "EP plane drift");
static_assert(kSmallPlaneCastlingWK == kPackPlaneCWK, "castling WK drift");
static_assert(kSmallPlaneHMC == kPackPlaneHMC, "HMC plane drift");

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

// RAII for an instantiated CUDA graph (the executable form). Holds both the
// captured cudaGraph_t and its cudaGraphExec_t; both are released on destroy.
class CudaGraphExec
{
public:
    CudaGraphExec() = default;
    ~CudaGraphExec() { reset(); }

    CudaGraphExec(const CudaGraphExec&) = delete;
    CudaGraphExec& operator=(const CudaGraphExec&) = delete;

    void reset()
    {
        if (mExec) { cudaGraphExecDestroy(mExec); mExec = nullptr; }
        if (mGraph) { cudaGraphDestroy(mGraph); mGraph = nullptr; }
    }

    // Capture the work submitted to `stream` by `submit()` and instantiate it.
    // Returns false (and leaves this empty) on any CUDA error so the caller
    // can fall back to the non-graph path.
    template <typename SubmitFn>
    bool capture(cudaStream_t stream, SubmitFn&& submit)
    {
        reset();
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
        {
            return false;
        }
        submit();
        cudaGraph_t graph = nullptr;
        if (cudaStreamEndCapture(stream, &graph) != cudaSuccess || graph == nullptr)
        {
            return false;
        }
        cudaGraphExec_t exec = nullptr;
        if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess || exec == nullptr)
        {
            cudaGraphDestroy(graph);
            return false;
        }
        mGraph = graph;
        mExec = exec;
        return true;
    }

    bool valid() const noexcept { return mExec != nullptr; }

    cudaError_t launch(cudaStream_t stream) const
    {
        return cudaGraphLaunch(mExec, stream);
    }

private:
    cudaGraph_t mGraph{nullptr};
    cudaGraphExec_t mExec{nullptr};
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

// Fill 12 piece planes in `features` starting at channel `base` for one
// position represented by its 12 piece bitboards (W:PNBRQK then B:PNBRQK).
// Convention matches the Python encoder: `tensor[7-rank, file, base+plane]`.
void fillPiecePlanesFromBitboards(
    std::vector<float>& features,
    const std::array<std::uint64_t, 12>& bbs,
    int base)
{
    for (int plane = 0; plane < kSmallPiecePlanes; ++plane)
    {
        std::uint64_t bb = bbs[static_cast<size_t>(plane)];
        while (bb != 0ULL)
        {
            const int squareIndex = __builtin_ctzll(bb);
            bb &= bb - 1;
            const int rank = squareIndex / kSmallBoardSize;
            const int file = squareIndex % kSmallBoardSize;
            features[encodeSmallIndex(kSmallBoardSize - 1 - rank, file, base + plane)] = 1.0F;
        }
    }
}

// Snapshot a chess::Board into the 12 piece bitboards (same order as the
// PiecePlanes struct). Inline duplicate of snapshot_piece_planes() in
// common.cpp — keeping it here avoids dragging that dependency into the
// inference TU and lets the encoder be self-contained.
std::array<std::uint64_t, 12> currentBoardBitboards(const chess::Board& board)
{
    std::array<std::uint64_t, 12> bbs{};
    constexpr std::array<chess::PieceType, 6> kTypes{
        chess::PieceType::PAWN,
        chess::PieceType::KNIGHT,
        chess::PieceType::BISHOP,
        chess::PieceType::ROOK,
        chess::PieceType::QUEEN,
        chess::PieceType::KING,
    };
    for (int t = 0; t < 6; ++t)
    {
        bbs[static_cast<size_t>(t)]     = board.pieces(kTypes[t], chess::Color::WHITE).getBits();
        bbs[static_cast<size_t>(t + 6)] = board.pieces(kTypes[t], chess::Color::BLACK).getBits();
    }
    return bbs;
}

std::vector<float> encodeSmallFeatures(
    const chess::Board& board,
    const chessrl::mcts::HistoryStack& history)
{
    std::vector<float> features(
        static_cast<size_t>(kSmallBoardSize) * kSmallBoardSize * kSmallFeaturePlanes, 0.0F);

    auto fillPlane = [&](int plane, float value) {
        for (int r = 0; r < kSmallBoardSize; ++r)
        {
            for (int c = 0; c < kSmallBoardSize; ++c)
            {
                features[encodeSmallIndex(r, c, plane)] = value;
            }
        }
    };

    // Slot 0: current position. We always use `board`'s actual bitboards
    // (rather than history[0]) so that a stale or incorrectly-maintained
    // ring buffer cannot poison the current planes.
    const auto current_bbs = currentBoardBitboards(board);
    fillPiecePlanesFromBitboards(features, current_bbs, /*base=*/0);
    if (history[0].repetition)
    {
        fillPlane(kSmallPiecePlanes, 1.0F);  // plane 12
    }

    // Slots 1..7: historical positions. base = 13 + (slot-1)*13.
    for (int slot = 1; slot < kSmallHistoryDepth; ++slot)
    {
        const auto& past = history[static_cast<size_t>(slot)];
        if (!past.valid)
        {
            continue;
        }
        const int base = kSmallPiecePlanes + 1 + (slot - 1) * kSmallHistBlock;
        fillPiecePlanesFromBitboards(features, past.bitboards, base);
        if (past.repetition)
        {
            fillPlane(base + kSmallPiecePlanes, 1.0F);
        }
    }

    // Globals (planes 104..110).
    fillPlane(kSmallPlaneSTM, board.sideToMove() == chess::Color::WHITE ? 1.0F : 0.0F);

    const auto enPassant = board.enpassantSq();
    if (enPassant.is_valid())
    {
        const int idxSq = enPassant.index();
        const int rank = idxSq / kSmallBoardSize;
        const int file = idxSq % kSmallBoardSize;
        features[encodeSmallIndex(kSmallBoardSize - 1 - rank, file, kSmallPlaneEP)] = 1.0F;
    }

    const auto castling = board.castlingRights();
    fillPlane(kSmallPlaneCastlingWK,
        castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
    fillPlane(kSmallPlaneCastlingWQ,
        castling.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);
    fillPlane(kSmallPlaneCastlingBK,
        castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE) ? 1.0F : 0.0F);
    fillPlane(kSmallPlaneCastlingBQ,
        castling.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE) ? 1.0F : 0.0F);

    const float halfMoveNorm = std::min(1.0F, static_cast<float>(board.halfMoveClock()) / 100.0F);
    fillPlane(kSmallPlaneHMC, halfMoveNorm);

    return features;
}

// Convenience wrapper: encode a board with no history (slots 1..7 zero,
// current repetition flag = 0). Used by `evaluateSmall` (single-shot
// inference helper).
std::vector<float> encodeSmallFeaturesNoHistory(const chess::Board& board)
{
    chessrl::mcts::HistoryStack empty{};
    empty[0].valid = true;  // slot 0 bitboards are overwritten from `board` anyway
    return encodeSmallFeatures(board, empty);
}

// ---------------------------------------------------------------------------
// Packed-input path (host-side reference).
//
// Instead of marshalling a dense 7104-float tensor per leaf (the measured
// ~0.78 ms/batch host-memory bottleneck), we carry a tiny ~776-byte packed
// snapshot and expand it to the 111 planes. expand_packed_to_dense() below
// is the CPU REFERENCE: it MUST be bit-identical to encodeSmallFeatures()
// for the same logical position. The GPU kernel (Step 2) mirrors this
// reference; the runtime path (Step 3) replaces the dense marshalling with
// packed-transfer + GPU expansion. Gated by tests/test_packed_parity.py.
// ---------------------------------------------------------------------------

void pack_small_position_impl(
    const chess::Board& board,
    const chessrl::mcts::HistoryStack& history,
    SmallPackedPosition& out)
{
    out = SmallPackedPosition{};

    // Slot 0 piece planes always come from the live board (matches
    // encodeSmallFeatures: defensive against a drifted ring buffer).
    const auto cur = currentBoardBitboards(board);
    for (int k = 0; k < kSmallPiecePlanes; ++k)
        out.bitboards[0][static_cast<size_t>(k)] = cur[static_cast<size_t>(k)];
    out.slot_valid = 0x01u;                                   // slot 0 always valid
    if (history[0].repetition) out.slot_rep |= 0x01u;         // current-rep flag

    for (int slot = 1; slot < kSmallHistoryDepth; ++slot)
    {
        const auto& past = history[static_cast<size_t>(slot)];
        if (!past.valid) continue;                            // zero slot ⇒ no-history
        for (int k = 0; k < kSmallPiecePlanes; ++k)
            out.bitboards[slot][static_cast<size_t>(k)] =
                past.bitboards[static_cast<size_t>(k)];
        out.slot_valid |= static_cast<std::uint8_t>(1u << slot);
        if (past.repetition)
            out.slot_rep |= static_cast<std::uint8_t>(1u << slot);
    }

    out.stm = (board.sideToMove() == chess::Color::WHITE) ? 1u : 0u;

    const auto ep = board.enpassantSq();
    out.ep_square = ep.is_valid()
        ? static_cast<std::int16_t>(ep.index())
        : static_cast<std::int16_t>(-1);

    const auto cr = board.castlingRights();
    std::uint8_t cb = 0;
    if (cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE))  cb |= 0x1u;
    if (cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE)) cb |= 0x2u;
    if (cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE))  cb |= 0x4u;
    if (cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE)) cb |= 0x8u;
    out.castling = cb;

    const auto hm = board.halfMoveClock();
    out.halfmove = static_cast<std::uint8_t>(hm > 255u ? 255u : hm);
}

// CPU reference expansion. Bit-identical to encodeSmallFeatures() — same
// plane indices, same rank-flip (row = 7 - rank), same globals. The GPU
// kernel must reproduce this exactly.
std::vector<float> expand_packed_to_dense_impl(const SmallPackedPosition& p)
{
    std::vector<float> features(
        static_cast<size_t>(kSmallBoardSize) * kSmallBoardSize * kSmallFeaturePlanes,
        0.0F);

    auto fillPlane = [&](int plane, float value) {
        for (int r = 0; r < kSmallBoardSize; ++r)
            for (int c = 0; c < kSmallBoardSize; ++c)
                features[encodeSmallIndex(r, c, plane)] = value;
    };
    auto fillPieces = [&](const std::uint64_t* bbs, int base) {
        for (int plane = 0; plane < kSmallPiecePlanes; ++plane)
        {
            std::uint64_t bb = bbs[plane];
            while (bb != 0ULL)
            {
                const int sq = __builtin_ctzll(bb);
                bb &= bb - 1;
                const int rank = sq / kSmallBoardSize;
                const int file = sq % kSmallBoardSize;
                features[encodeSmallIndex(kSmallBoardSize - 1 - rank, file, base + plane)] = 1.0F;
            }
        }
    };

    // Slot 0 (always valid).
    fillPieces(p.bitboards[0], 0);
    if (p.slot_rep & 0x01u) fillPlane(kSmallPiecePlanes, 1.0F);

    for (int slot = 1; slot < kSmallHistoryDepth; ++slot)
    {
        if (!((p.slot_valid >> slot) & 1u)) continue;
        const int base = kSmallPiecePlanes + 1 + (slot - 1) * kSmallHistBlock;
        fillPieces(p.bitboards[slot], base);
        if ((p.slot_rep >> slot) & 1u) fillPlane(base + kSmallPiecePlanes, 1.0F);
    }

    fillPlane(kSmallPlaneSTM, p.stm ? 1.0F : 0.0F);

    if (p.ep_square >= 0)
    {
        const int idxSq = p.ep_square;
        const int rank = idxSq / kSmallBoardSize;
        const int file = idxSq % kSmallBoardSize;
        features[encodeSmallIndex(kSmallBoardSize - 1 - rank, file, kSmallPlaneEP)] = 1.0F;
    }

    fillPlane(kSmallPlaneCastlingWK, (p.castling & 0x1u) ? 1.0F : 0.0F);
    fillPlane(kSmallPlaneCastlingWQ, (p.castling & 0x2u) ? 1.0F : 0.0F);
    fillPlane(kSmallPlaneCastlingBK, (p.castling & 0x4u) ? 1.0F : 0.0F);
    fillPlane(kSmallPlaneCastlingBQ, (p.castling & 0x8u) ? 1.0F : 0.0F);

    const float hmNorm = std::min(1.0F, static_cast<float>(p.halfmove) / 100.0F);
    fillPlane(kSmallPlaneHMC, hmNorm);

    return features;
}

SmallEncodedPosition build_small_encoded_position(
    const chess::Board& board,
    const chessrl::mcts::HistoryStack& history)
{
    const auto t0 = std::chrono::steady_clock::now();
    SmallEncodedPosition encoded;
    encoded.features = encodeSmallFeatures(board, history);
    const auto t1 = std::chrono::steady_clock::now();
    g_encoder_calls.fetch_add(1, std::memory_order_relaxed);
    g_encoder_nanos.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
        std::memory_order_relaxed);
    return encoded;
}

SmallEncodedPosition build_small_encoded_position(const chess::Board& board)
{
    const auto t0 = std::chrono::steady_clock::now();
    SmallEncodedPosition encoded;
    encoded.features = encodeSmallFeaturesNoHistory(board);
    const auto t1 = std::chrono::steady_clock::now();
    g_encoder_calls.fetch_add(1, std::memory_order_relaxed);
    g_encoder_nanos.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
        std::memory_order_relaxed);
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

    // Defensive: refuse to run if the loaded TRT engine's input dim doesn't
    // match kSmallFeaturePlanes. Catches stale engines built against the
    // legacy 19-plane layout (the exact bug class we just suffered with the
    // cache rank-flip — silent decode mismatch).
    {
        const auto profile_max = engine.getProfileShape(
            ctx.inputName().c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        if (profile_max.nbDims >= 4
            && profile_max.d[3] != kSmallFeaturePlanes)
        {
            throw std::runtime_error(
                "TRT engine input channels (" + std::to_string(profile_max.d[3])
                + ") != expected kSmallFeaturePlanes ("
                + std::to_string(kSmallFeaturePlanes)
                + "). Rebuild the engine with the matching layout.");
        }
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

    // No-history one-shot eval: caller is responsible for passing the FEN of
    // the position they care about. Historical slots stay zero.
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

SmallEncodedPosition encode_small_position(
    const chess::Board& board,
    const chessrl::mcts::HistoryStack& history)
{
    return build_small_encoded_position(board, history);
}

SmallPackedPosition pack_small_position(
    const chess::Board& board, const chessrl::mcts::HistoryStack& history)
{
    SmallPackedPosition p;
    pack_small_position_impl(board, history, p);
    return p;
}

std::vector<float> expand_packed_to_dense(const SmallPackedPosition& packed)
{
    return expand_packed_to_dense_impl(packed);
}

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
        std::unique_ptr<nvinfer1::IExecutionContext> exec;  // dynamic-shape, fallback path
        std::unique_ptr<CudaStream> stream;
        ResizableDeviceBuffer packedDevice;  // ~776 B/leaf, H2D'd then expanded
        ResizableDeviceBuffer inputDevice;   // dense fp16 [B,8,8,111], kernel out
        ResizableDeviceBuffer policyDevice;
        ResizableDeviceBuffer valueDevice;
        ResizableDeviceBuffer mlhDevice;
        ResizableDeviceBuffer mlhQDevice;
        PinnedBuffer hostPacked;             // packed snapshots staged for H2D
        PinnedBuffer hostPolicy;
        PinnedBuffer hostValue;
        PinnedBuffer hostMlh;
        PinnedBuffer hostMlhQ;
        // CUDA-graph state. One frozen-shape IExecutionContext + one captured
        // graph per bucket. The contexts are shape-pinned (created once at
        // their bucket size, never reshaped) so the kernels/internal device
        // memory the graph captured stay valid across replays. Buckets that
        // fail capture go in graph_failed and permanently use the fallback.
        std::unordered_map<int, std::unique_ptr<nvinfer1::IExecutionContext>> graph_ctx;
        std::unordered_map<int, CudaGraphExec> graphs;
        std::unordered_set<int> graph_failed;
        bool buffers_sized_for_graphs{false};
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
        // CUDA-graph config. Default ON; CHESSRL_CUDA_GRAPHS=0 disables
        // (instant rollback, no rebuild). CHESSRL_GRAPH_BUCKETS overrides the
        // ascending bucket list.
        if (const char* env = std::getenv("CHESSRL_CUDA_GRAPHS"))
        {
            graphs_enabled_ = (std::atoi(env) != 0);
        }
        {
            std::string spec = "32,64,128,256";
            if (const char* env = std::getenv("CHESSRL_GRAPH_BUCKETS"))
            {
                if (env[0] != '\0') spec = env;
            }
            std::stringstream ss(spec);
            std::string tok;
            while (std::getline(ss, tok, ','))
            {
                const int b = std::atoi(tok.c_str());
                if (b > 0) graph_buckets_.push_back(b);
            }
            std::sort(graph_buckets_.begin(), graph_buckets_.end());
            graph_buckets_.erase(
                std::unique(graph_buckets_.begin(), graph_buckets_.end()),
                graph_buckets_.end());
            // Guarantee the top bucket covers any batch the queue can hand us,
            // so the device/pinned buffers are sized once (to the top bucket)
            // and never reallocated — captured graph pointers stay valid.
            if (graph_buckets_.empty()
                || graph_buckets_.back() < config.max_batch_size)
            {
                graph_buckets_.push_back(config.max_batch_size);
            }
            // Drop buckets larger than max_batch_size except the (single)
            // covering one — they can never be selected.
            while (graph_buckets_.size() >= 2
                   && graph_buckets_[graph_buckets_.size() - 2] >= config.max_batch_size)
            {
                graph_buckets_.pop_back();
            }
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
        if (telemetry_enabled_)
        {
            const auto g = g_graph_batches.load(std::memory_order_relaxed);
            const auto f = g_fallback_batches.load(std::memory_order_relaxed);
            if (g + f > 0)
            {
                const double n = static_cast<double>(g + f);
                std::cerr << "[cuda-graph] batches: graph=" << g
                          << " fallback=" << f << " ("
                          << (100.0 * static_cast<double>(g) / n)
                          << "% on graphs)\n";
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
            
            // ---- Path + engine batch -------------------------------------
            // Graph path: smallest bucket >= actualCount, fixed-shape captured
            // graph (engineBatch = bucket, pad rows zero-filled). Fallback:
            // dynamic shape, engineBatch = actualCount, original enqueueV3.
            const bool use_graph = graphs_enabled_ && !graph_buckets_.empty();
            int bucket = 0;
            if (use_graph)
            {
                auto it = std::lower_bound(
                    graph_buckets_.begin(), graph_buckets_.end(),
                    static_cast<int>(actualCount));
                bucket = (it == graph_buckets_.end())
                    ? graph_buckets_.back() : *it;
            }
            const size_t engineBatch = use_graph
                ? static_cast<size_t>(bucket) : actualCount;
            // Size buffers to the TOP bucket when graphs are on so device /
            // pinned pointers never move after a graph captured them
            // (ResizableDeviceBuffer::ensure only ever grows).
            const size_t sizeBatch = use_graph
                ? static_cast<size_t>(graph_buckets_.back()) : actualCount;

            inputDims.d[0] = static_cast<int>(engineBatch);

            // packedDevice/hostPacked: the compact ~776 B/leaf snapshots.
            // inputDevice: the dense [B,8,8,111] tensor the GPU kernel writes
            // and TRT reads (no host dense tensor exists anymore).
            const size_t packedBytes = engineBatch * sizeof(SmallPackedPosition);
            w.packedDevice.ensure(sizeBatch * sizeof(SmallPackedPosition));
            w.hostPacked.ensure(sizeBatch * sizeof(SmallPackedPosition));
            w.inputDevice.ensure(sizeBatch * inputElements * element_size(inputType));
            w.policyDevice.ensure(sizeBatch * policyElements * element_size(policyType));
            w.valueDevice.ensure(sizeBatch * valueElements * element_size(valueType));
            const size_t mlhBytes = hasMlh
                ? engineBatch * mlhElements * element_size(mlhType)
                : 0;
            const size_t mlhQBytes = hasMlhQ
                ? engineBatch * mlhElements * element_size(mlhQType)
                : 0;
            if (hasMlh) w.mlhDevice.ensure(sizeBatch * mlhElements * element_size(mlhType));
            if (hasMlhQ) w.mlhQDevice.ensure(sizeBatch * mlhElements * element_size(mlhQType));

            const size_t policyBytes = engineBatch * policyElements * element_size(policyType);
            const size_t valueBytes = engineBatch * valueElements * element_size(valueType);
            w.hostPolicy.ensure(sizeBatch * policyElements * element_size(policyType));
            w.hostValue.ensure(sizeBatch * valueElements * element_size(valueType));
            if (hasMlh) w.hostMlh.ensure(sizeBatch * mlhElements * element_size(mlhType));
            if (hasMlhQ) w.hostMlhQ.ensure(sizeBatch * mlhElements * element_size(mlhQType));
            const int out_is_half = (inputType == nvinfer1::DataType::kHALF) ? 1 : 0;

            // Step 3: Input assembly is now a flat memcpy of the compact
            // ~776 B packed snapshots into the pinned staging buffer (the
            // ~0.78 ms scalar-fp16-conversion of 28 KB dense tensors is
            // gone). Pad rows [actualCount,engineBatch) are zeroed: a zeroed
            // SmallPackedPosition has slot_valid=0 ⇒ the kernel emits all-
            // zero planes for it. Pad outputs are discarded, but a captured
            // graph copies the full bucket region every replay so the pad
            // must be deterministic.
            {
                auto* hp = static_cast<SmallPackedPosition*>(w.hostPacked.data());
                for (size_t i = 0; i < actualCount; ++i)
                    hp[i] = *work[i].position;  // 776 B copy, per-batch (NOT under the queue lock)
                if (engineBatch > actualCount)
                    std::memset(hp + actualCount, 0,
                        (engineBatch - actualCount) * sizeof(SmallPackedPosition));
            }

            auto bind_addresses = [&](nvinfer1::IExecutionContext* ex) -> bool {
                bool ok = ex->setTensorAddress(ctx.inputName().c_str(), w.inputDevice.data())
                    && ex->setTensorAddress(ctx.policyName().c_str(), w.policyDevice.data())
                    && ex->setTensorAddress(ctx.valueName().c_str(), w.valueDevice.data());
                if (ok && hasMlh)
                    ok = ex->setTensorAddress(ctx.mlhName().c_str(), w.mlhDevice.data());
                if (ok && hasMlhQ)
                    ok = ex->setTensorAddress(ctx.mlhQName().c_str(), w.mlhQDevice.data());
                return ok;
            };

            // H2D the packed snapshots, then expand them to the dense fp16
            // input tensor on the GPU. Both ops go on `s` so the whole thing
            // is captured inside a CUDA graph. Replaces the old dense H2D.
            auto submit_input = [&](cudaStream_t s) -> bool {
                if (cudaMemcpyAsync(w.packedDevice.data(), w.hostPacked.data(),
                        packedBytes, cudaMemcpyHostToDevice, s) != cudaSuccess)
                    return false;
                return launch_expand_planes(
                    static_cast<const SmallPackedPosition*>(w.packedDevice.data()),
                    w.inputDevice.data(), static_cast<int>(engineBatch),
                    out_is_half, s) == cudaSuccess;
            };


            if (use_graph)
            {
                // Per-bucket frozen-shape context: created once, shape and
                // tensor addresses fixed forever (buffers were sized to the
                // top bucket above so the device pointers never move).
                auto& gctx = w.graph_ctx[bucket];
                if (!gctx)
                {
                    gctx.reset(ctx.engine().createExecutionContext());
                    if (!gctx)
                        throw std::runtime_error("Failed to create graph exec context.");
                    nvinfer1::Dims gd = engine.getTensorShape(ctx.inputName().c_str());
                    gd.d[0] = bucket;
                    if (!gctx->setInputShape(ctx.inputName().c_str(), gd))
                        throw std::runtime_error("Failed to set graph context input shape.");
                    if (!bind_addresses(gctx.get()))
                        throw std::runtime_error("Failed to bind graph context addresses.");
                }

                CudaGraphExec& g = w.graphs[bucket];
                const bool failed = w.graph_failed.count(bucket) != 0;
                if (!g.valid() && !failed)
                {
                    // Warmup OUTSIDE capture so TRT finalizes internal device
                    // allocations / tactic selection for this shape.
                    if (!gctx->enqueueV3(w.stream->get()))
                        throw std::runtime_error("Graph warmup enqueueV3 failed.");
                    if (cudaStreamSynchronize(w.stream->get()) != cudaSuccess)
                        throw std::runtime_error("Graph warmup sync failed.");

                    bool enqueue_ok = true;
                    const bool captured = g.capture(w.stream->get(), [&]() {
                        if (!submit_input(w.stream->get())) enqueue_ok = false;
                        if (!gctx->enqueueV3(w.stream->get())) enqueue_ok = false;
                        cudaMemcpyAsync(w.hostPolicy.data(), w.policyDevice.data(),
                            policyBytes, cudaMemcpyDeviceToHost, w.stream->get());
                        cudaMemcpyAsync(w.hostValue.data(), w.valueDevice.data(),
                            valueBytes, cudaMemcpyDeviceToHost, w.stream->get());
                        if (hasMlh)
                            cudaMemcpyAsync(w.hostMlh.data(), w.mlhDevice.data(),
                                mlhBytes, cudaMemcpyDeviceToHost, w.stream->get());
                        if (hasMlhQ)
                            cudaMemcpyAsync(w.hostMlhQ.data(), w.mlhQDevice.data(),
                                mlhQBytes, cudaMemcpyDeviceToHost, w.stream->get());
                    });
                    if (!captured || !enqueue_ok)
                    {
                        g.reset();
                        w.graph_failed.insert(bucket);
                        std::cerr << "[cuda-graph] capture failed for bucket "
                                  << bucket << " — using enqueueV3 fallback\n";
                    }
                }

                if (g.valid())
                {
                    if (g.launch(w.stream->get()) != cudaSuccess)
                        throw std::runtime_error("cudaGraphLaunch failed.");
                    if (cudaStreamSynchronize(w.stream->get()) != cudaSuccess)
                        throw std::runtime_error("cudaStreamSynchronize (graph) failed.");
                    g_graph_batches.fetch_add(1, std::memory_order_relaxed);
                }
                else
                {
                    // Capture unavailable: run the same frozen context directly.
                    if (!submit_input(w.stream->get()))
                        throw std::runtime_error("submit_input (gfb) failed.");
                    if (!gctx->enqueueV3(w.stream->get()))
                        throw std::runtime_error("graph-fallback enqueueV3 failed.");
                    if (cudaMemcpyAsync(w.hostPolicy.data(), w.policyDevice.data(),
                            policyBytes, cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
                        throw std::runtime_error("cudaMemcpyAsync (gfb policy) failed.");
                    if (cudaMemcpyAsync(w.hostValue.data(), w.valueDevice.data(),
                            valueBytes, cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
                        throw std::runtime_error("cudaMemcpyAsync (gfb value) failed.");
                    if (hasMlh && cudaMemcpyAsync(w.hostMlh.data(), w.mlhDevice.data(),
                            mlhBytes, cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
                        throw std::runtime_error("cudaMemcpyAsync (gfb mlh) failed.");
                    if (hasMlhQ && cudaMemcpyAsync(w.hostMlhQ.data(), w.mlhQDevice.data(),
                            mlhQBytes, cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
                        throw std::runtime_error("cudaMemcpyAsync (gfb mlh_q) failed.");
                    if (cudaStreamSynchronize(w.stream->get()) != cudaSuccess)
                        throw std::runtime_error("cudaStreamSynchronize (gfb) failed.");
                    g_fallback_batches.fetch_add(1, std::memory_order_relaxed);
                }
            }

            if (!use_graph)
            {
                // Original dynamic-shape path (graphs disabled).
                if (!w.exec->setInputShape(ctx.inputName().c_str(), inputDims))
                    throw std::runtime_error("Failed to set small-model batch input shape.");
                if (!submit_input(w.stream->get()))
                    throw std::runtime_error("submit_input (no-graph) failed.");
                if (!bind_addresses(w.exec.get()))
                    throw std::runtime_error("Failed to set small-model batch tensor addresses.");
                if (!w.exec->enqueueV3(w.stream->get()))
                    throw std::runtime_error("Small-model batched TensorRT enqueue failed.");
                if (cudaMemcpyAsync(w.hostPolicy.data(), w.policyDevice.data(), policyBytes,
                        cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
                    throw std::runtime_error("cudaMemcpyAsync (small batch policy D2H) failed.");
                if (cudaMemcpyAsync(w.hostValue.data(), w.valueDevice.data(), valueBytes,
                        cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
                    throw std::runtime_error("cudaMemcpyAsync (small batch value D2H) failed.");
                if (hasMlh && cudaMemcpyAsync(w.hostMlh.data(), w.mlhDevice.data(), mlhBytes,
                        cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
                    throw std::runtime_error("cudaMemcpyAsync (small batch mlh D2H) failed.");
                if (hasMlhQ && cudaMemcpyAsync(w.hostMlhQ.data(), w.mlhQDevice.data(), mlhQBytes,
                        cudaMemcpyDeviceToHost, w.stream->get()) != cudaSuccess)
                    throw std::runtime_error("cudaMemcpyAsync (small batch mlh_q D2H) failed.");
                if (cudaStreamSynchronize(w.stream->get()) != cudaSuccess)
                    throw std::runtime_error("cudaStreamSynchronize (small batch) failed.");
                g_fallback_batches.fetch_add(1, std::memory_order_relaxed);
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

    // CUDA-graph capture config. graphs_enabled_ is gated by
    // CHESSRL_CUDA_GRAPHS (default on). graph_buckets_ is the ascending set
    // of fixed batch shapes we capture; the largest is always >=
    // max_batch_size so a bucket always covers any drained batch (no
    // size-based fallback that would force a device-buffer realloc and
    // invalidate captured pointers).
    bool graphs_enabled_{true};
    std::vector<int> graph_buckets_;
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

