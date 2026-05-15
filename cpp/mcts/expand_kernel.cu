// Packed → dense plane expansion kernel. Mirrors expand_packed_to_dense()
// (CPU reference, single_inference.cpp) bit-for-bit: same plane indices,
// same 7-rank flip (row = 7 - rank), same globals. Validated GPU-vs-CPU by
// tests/test_packed_gpu_parity.py.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mcts/expand_kernel.hpp"

namespace chessrl::mcts::single
{
namespace
{

__device__ __forceinline__ void store(float* p, float v) { *p = v; }
__device__ __forceinline__ void store(__half* p, float v) { *p = __float2half(v); }

// dense index: b*7104 + (row*8 + col)*111 + plane
__device__ __forceinline__ int didx(int b, int row, int col, int plane)
{
    return b * kPackElems + (row * kPackBoard + col) * kPackPlanes + plane;
}

template <typename T>
__global__ void expand_kernel(const SmallPackedPosition* __restrict__ packed,
                              T* __restrict__ out, int batch)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const SmallPackedPosition& p = packed[b];

    auto fill_plane = [&](int plane, float v) {
        for (int r = 0; r < kPackBoard; ++r)
            for (int c = 0; c < kPackBoard; ++c)
                store(&out[didx(b, r, c, plane)], v);
    };
    auto fill_pieces = [&](const unsigned long long* bbs, int base) {
        for (int k = 0; k < kPackPiecePlanes; ++k)
        {
            unsigned long long bb = bbs[k];
            while (bb != 0ULL)
            {
                const int sq = __ffsll(static_cast<long long>(bb)) - 1;
                bb &= bb - 1;
                const int rank = sq / kPackBoard;
                const int file = sq % kPackBoard;
                store(&out[didx(b, kPackBoard - 1 - rank, file, base + k)], 1.0f);
            }
        }
    };

    // Slot 0 (always valid).
    fill_pieces(reinterpret_cast<const unsigned long long*>(p.bitboards[0]), 0);
    if (p.slot_rep & 0x01u) fill_plane(kPackPiecePlanes, 1.0f);

    for (int slot = 1; slot < kPackHistory; ++slot)
    {
        if (!((p.slot_valid >> slot) & 1u)) continue;
        const int base = kPackPiecePlanes + 1 + (slot - 1) * kPackHistBlock;
        fill_pieces(reinterpret_cast<const unsigned long long*>(p.bitboards[slot]), base);
        if ((p.slot_rep >> slot) & 1u) fill_plane(base + kPackPiecePlanes, 1.0f);
    }

    if (p.stm) fill_plane(kPackPlaneSTM, 1.0f);  // 0 already from memset

    if (p.ep_square >= 0)
    {
        const int sq = p.ep_square;
        store(&out[didx(b, kPackBoard - 1 - sq / kPackBoard, sq % kPackBoard,
                        kPackPlaneEP)], 1.0f);
    }

    if (p.castling & 0x1u) fill_plane(kPackPlaneCWK, 1.0f);
    if (p.castling & 0x2u) fill_plane(kPackPlaneCWQ, 1.0f);
    if (p.castling & 0x4u) fill_plane(kPackPlaneCBK, 1.0f);
    if (p.castling & 0x8u) fill_plane(kPackPlaneCBQ, 1.0f);

    const float hm = static_cast<float>(p.halfmove) / 100.0f;
    fill_plane(kPackPlaneHMC, hm < 1.0f ? hm : 1.0f);
}

} // namespace

cudaError_t launch_expand_planes(
    const SmallPackedPosition* d_packed,
    void* d_out,
    int batch,
    int out_is_half,
    cudaStream_t stream)
{
    if (batch <= 0) return cudaSuccess;

    const size_t elem = out_is_half ? sizeof(__half) : sizeof(float);
    cudaError_t e = cudaMemsetAsync(
        d_out, 0, static_cast<size_t>(batch) * kPackElems * elem, stream);
    if (e != cudaSuccess) return e;

    const int threads = 128;
    const int blocks = (batch + threads - 1) / threads;
    if (out_is_half)
        expand_kernel<__half><<<blocks, threads, 0, stream>>>(
            d_packed, static_cast<__half*>(d_out), batch);
    else
        expand_kernel<float><<<blocks, threads, 0, stream>>>(
            d_packed, static_cast<float*>(d_out), batch);
    return cudaGetLastError();
}

} // namespace chessrl::mcts::single
