#pragma once

// Layout of the network's 8 x 8 x 111 input and of its outputs, shared by the CPU
// encoders (small_encoding.cpp) and the network runner (single_inference.cpp).

#include "mcts/packed_position.hpp"

namespace chessrl::mcts::single
{

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

} // namespace chessrl::mcts::single
