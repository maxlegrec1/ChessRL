#pragma once

// Standalone POD definition shared by the host encoder (g++ TU) and the
// CUDA expansion kernel (nvcc TU). Deliberately depends ONLY on <cstdint>
// so nvcc never has to parse chess-library headers.

#include <cstdint>

namespace chessrl::mcts::single
{

// Compact per-leaf snapshot (~776 B) replacing the dense 7104-float tensor
// on the host→queue→H2D path. Trivially copyable POD. slot 0 = current
// position. See expand_packed_to_dense() for the exact plane semantics.
//   bitboards[s][k]: slot s, piece plane k (W:PNBRQK then B:PNBRQK)
//   slot_valid : bit s ⇒ slot s piece planes present (bit 0 always set)
//   slot_rep   : bit s ⇒ slot s repetition plane lit (bit 0 = current-rep)
//   stm        : 1 if white is to move in the real position (kept by mirror_to_white: a canonical position
//                with stm 0 had black to move; plane 104 is stm or 1, per the network's stm_colour)
//   ep_square  : 0..63 board square index, or -1 if none
//   castling   : bit0 WK, bit1 WQ, bit2 BK, bit3 BQ
//   halfmove   : raw clock; expansion emits min(1, halfmove/100)
struct SmallPackedPosition
{
    std::uint64_t bitboards[8][12]{};
    std::uint8_t  slot_valid{0};
    std::uint8_t  slot_rep{0};
    std::uint8_t  stm{0};
    std::int16_t  ep_square{-1};
    std::uint8_t  castling{0};
    std::uint8_t  halfmove{0};
};

// Plane-layout constants (single source of truth, shared host/device).
constexpr int kPackBoard       = 8;
constexpr int kPackPiecePlanes = 12;
constexpr int kPackHistory     = 8;
constexpr int kPackHistBlock   = kPackPiecePlanes + 1;            // 13
constexpr int kPackGlobals     = 7;
constexpr int kPackPlanes      = kPackHistory * kPackHistBlock + kPackGlobals; // 111
constexpr int kPackPlaneSTM    = kPackHistory * kPackHistBlock;   // 104
constexpr int kPackPlaneEP     = kPackPlaneSTM + 1;               // 105
constexpr int kPackPlaneCWK    = kPackPlaneSTM + 2;               // 106
constexpr int kPackPlaneCWQ    = kPackPlaneSTM + 3;               // 107
constexpr int kPackPlaneCBK    = kPackPlaneSTM + 4;               // 108
constexpr int kPackPlaneCBQ    = kPackPlaneSTM + 5;               // 109
constexpr int kPackPlaneHMC    = kPackPlaneSTM + 6;               // 110
constexpr int kPackElems       = kPackBoard * kPackBoard * kPackPlanes; // 7104

// The canonical view: the network always plays white, so a position with black to move is seen mirrored, ranks
// reversed and white's and black's pieces and castling rights swapped (python-chess Board.mirror(),
// src/models/small/fen_encoder.py). Squares are a1 = 0 .. h8 = 63.
inline int mirror_square(int sq) { return sq ^ 56; }

// 12 piece bitboards, W:PNBRQK then B:PNBRQK, in the other side's view.
inline void mirror_bitboards(std::uint64_t* bbs)
{
    for (int k = 0; k < 6; ++k)
    {
        const std::uint64_t white = bbs[k];
        bbs[k] = __builtin_bswap64(bbs[k + 6]);
        bbs[k + 6] = __builtin_bswap64(white);
    }
}

// Castling rights WK = 1, WQ = 2, BK = 4, BQ = 8 in the other side's view.
inline std::uint8_t mirror_castling(std::uint8_t c)
{
    return static_cast<std::uint8_t>(((c >> 2) & 0x3u) | ((c & 0x3u) << 2));
}

// A packed position with black to move as white sees it: every history slot mirrored. stm keeps the real side
// to move (0), which networks with stm_colour read from plane 104; the others see 1 there.
inline void mirror_to_white(SmallPackedPosition& p)
{
    for (auto& slot : p.bitboards) mirror_bitboards(slot);
    if (p.ep_square >= 0) p.ep_square = static_cast<std::int16_t>(mirror_square(p.ep_square));
    p.castling = mirror_castling(p.castling);
}

// Input plane 104: the real side to move for networks trained with it (model.stm_plane colour), else 1.
inline float stm_plane_value(const SmallPackedPosition& p, bool stm_colour)
{
    return (!stm_colour || p.stm) ? 1.0F : 0.0F;
}

} // namespace chessrl::mcts::single
