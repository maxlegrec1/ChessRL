// CPU side of the network input: the dense 8 x 8 x 111 encoder, the packed snapshot a
// search leaf travels as, and its CPU expansion (the reference for expand_kernel.cu).
// No TensorRT or CUDA here, so the search builds and runs without a GPU
// (scripts/build_cpu_bench.sh).
#include "mcts/single_inference.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

#include "chess-library/include/chess.hpp"
#include "mcts/small_layout.hpp"

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

} // namespace chessrl::mcts::single
