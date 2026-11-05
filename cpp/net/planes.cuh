#pragma once

// The network's input planes from a packed position, on the GPU: shared by the fused kernel (encoder_kernel.cu) and
// the wide path (encoder_wide.cu), so both read positions exactly as expand_packed_to_dense (small_encoding.cpp).

#include <cuda_fp16.h>

#include <cstdint>

#include "mcts/packed_position.hpp"

namespace chessrl::net
{

// The input planes of token `cell` (row-major 8x8, row 0 = rank 8) as expand_packed_to_dense
// (small_encoding.cpp) builds them, FP16 into `row`; planes 111..127 are the zero padding of the
// embedding's k loop. Four threads per token (q = 0..3, adjacent lanes): each builds the 13-bit fields of history
// slots 2q and 2q + 1 (12 piece planes + repetition, zero for a slot past the game's start), the four OR them into the
// token's 128-bit plane mask, and each writes planes 32q..32q + 31 as four 16-byte stores (a per-plane switch, 32 per
// thread, was 12.7% of the fused kernel at 4 layers). Positions arrive canonical (pack_small_position mirrors those
// with black to move); only plane 104 can depend on the colour: p.stm (the real side to move) for networks with
// stm_colour, 1 for the others (stm_plane_value in packed_position.hpp). Called by all 32 lanes of the warp (the OR
// uses shuffles).
static_assert(mcts::single::kPackHistory == 8 && mcts::single::kPackPlaneHMC - 96 == 14,
              "two slots per thread, the halfmove plane in word 3");
__device__ __forceinline__ unsigned plane_word(const mcts::single::SmallPackedPosition& p, int cell, int q,
                                               int stm_colour)
{
    using namespace mcts::single;
    const int sq = (kPackBoard - 1 - (cell >> 3)) * kPackBoard + (cell & 7);
    unsigned long long lo = 0, hi = 0;  // planes 0..63, 64..127
#pragma unroll
    for (int j = 0; j < 2; ++j)
    {
        const int slot = 2 * q + j;
        if (slot != 0 && !((p.slot_valid >> slot) & 1u)) continue;
        unsigned long long f = static_cast<unsigned long long>((p.slot_rep >> slot) & 1u) << kPackPiecePlanes;
#pragma unroll
        for (int w = 0; w < kPackPiecePlanes; ++w) f |= ((p.bitboards[slot][w] >> sq) & 1ull) << w;
        const int o = slot * kPackHistBlock;
        if (o < 64)
        {
            lo |= f << o;
            if (o + kPackHistBlock > 64) hi |= f >> (64 - o);
        }
        else
        {
            hi |= f << (o - 64);
        }
    }
    lo |= __shfl_xor_sync(0xffffffffu, lo, 1);
    hi |= __shfl_xor_sync(0xffffffffu, hi, 1);
    lo |= __shfl_xor_sync(0xffffffffu, lo, 2);
    hi |= __shfl_xor_sync(0xffffffffu, hi, 2);
    hi |= static_cast<unsigned long long>(!stm_colour || p.stm) << (kPackPlaneSTM - 64);
    hi |= static_cast<unsigned long long>(p.ep_square == sq) << (kPackPlaneEP - 64);
    hi |= static_cast<unsigned long long>(p.castling & 0xFu) << (kPackPlaneCWK - 64);  // WK, WQ, BK, BQ
    return static_cast<unsigned>((q < 2 ? lo : hi) >> (32 * (q & 1)));
}

// Bit j of plane_word = plane 32q + j: as FP16, 1.0 for set planes and the halfmove plane's clock / 100.
__device__ __forceinline__ void planes_to_row(const mcts::single::SmallPackedPosition& p, int cell, int q,
                                              int stm_colour, __half* row)
{
    const unsigned word = plane_word(p, cell, q, stm_colour);
    uint32_t hv[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)  // planes 32q + 2j and + 1 as a half2 of 0.0 / 1.0 (0x3C00)
        hv[j] = ((word >> (2 * j)) & 1u) * 0x3C00u | ((word >> (2 * j + 1)) & 1u) * 0x3C000000u;
    if (q == 3)  // plane 110: the halfmove clock / 100, at most 1 (plane 111 is padding)
        hv[(mcts::single::kPackPlaneHMC - 96) / 2] =
            __half_as_ushort(__float2half(fminf(static_cast<float>(p.halfmove) / 100.f, 1.f)));
    uint4* dst = reinterpret_cast<uint4*>(row + 32 * q);
#pragma unroll
    for (int k = 0; k < 4; ++k) dst[k] = make_uint4(hv[4 * k], hv[4 * k + 1], hv[4 * k + 2], hv[4 * k + 3]);
}

// The same as int8 for an int8 network's embedding (src/export/int8.py, PLANE_SCALE): 100 for set planes and the
// halfmove clock (at most 100) for plane 110, exact at the row scale 1/100.
__device__ __forceinline__ void planes_to_row_i8(const mcts::single::SmallPackedPosition& p, int cell, int q,
                                                 int stm_colour, int8_t* row)
{
    const unsigned word = plane_word(p, cell, q, stm_colour);
    uint32_t o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)  // planes 32q + 4j..+3: each bit of the nibble to its byte's low bit, times 100
        o[j] = ((((word >> (4 * j)) & 0xFu) * 0x00204081u) & 0x01010101u) * 100u;
    if (q == 3)  // plane 110 = byte 14 of the thread's 32 (its bit is clear)
        o[3] |= static_cast<uint32_t>(min(static_cast<int>(p.halfmove), 100)) << 16;
    uint4* dst = reinterpret_cast<uint4*>(row + 32 * q);
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

} // namespace chessrl::net
