#pragma once

// Standalone POD definition shared by the host encoder (g++ TU) and the
// CUDA expansion kernel (nvcc TU). Deliberately depends ONLY on <cstdint>
// so nvcc never has to parse chess-library / TensorRT headers.

#include <cstdint>

namespace chessrl::mcts::single
{

// Compact per-leaf snapshot (~776 B) replacing the dense 7104-float tensor
// on the host→queue→H2D path. Trivially copyable POD. slot 0 = current
// position. See expand_packed_to_dense() for the exact plane semantics.
//   bitboards[s][k]: slot s, piece plane k (W:PNBRQK then B:PNBRQK)
//   slot_valid : bit s ⇒ slot s piece planes present (bit 0 always set)
//   slot_rep   : bit s ⇒ slot s repetition plane lit (bit 0 = current-rep)
//   stm        : 1 if white to move
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

} // namespace chessrl::mcts::single
