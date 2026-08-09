// Packed → dense plane expansion. Mirrors expand_packed_to_dense() (CPU
// reference, single_inference.cpp) bit-for-bit. Validated GPU-vs-CPU by
// tests/test_packed_gpu_parity.py.
//
// Design: ONE THREAD PER OUTPUT ELEMENT (batch * 7104). Every element is
// written exactly once (no memset), and consecutive threads write
// consecutive memory (the dense layout is plane-contiguous), so stores are
// fully coalesced. This replaces a naive one-thread-per-sample kernel that
// did ~1000 scattered writes/sample + a 3.6 MB memset and was slower than
// the host marshalling it removed.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mcts/expand_kernel.hpp"

namespace chessrl::mcts::single
{
namespace
{

__device__ __forceinline__ void store(float* p, float v) { *p = v; }
__device__ __forceinline__ void store(__half* p, float v) { *p = __float2half(v); }

// 3D launch — NO integer division on the hot path (the flat-index version
// did 4 div/mod by 111/8 per thread → ~25× slower than memory-bound):
//   blockIdx.x  = sample b      (gridDim.x  = batch)
//   blockIdx.y  = cell 0..63    (gridDim.y  = 64)
//   threadIdx.x = plane 0..110  (blockDim.x = 128; planes 111..127 idle)
// row/col from cell via >>3 / &7. Output index is one mul-add.
template <typename T>
__global__ void expand_kernel(const SmallPackedPosition* __restrict__ packed,
                              T* __restrict__ out)
{
    const int plane = threadIdx.x;
    if (plane >= kPackPlanes) return;
    const int b = blockIdx.x;
    const int cell = blockIdx.y;          // 0..63
    const int row = cell >> 3;            // /8
    const int col = cell & 7;             // %8
    const long long idx =
        ((long long)b * (kPackBoard * kPackBoard) + cell) * kPackPlanes + plane;

    {
        const SmallPackedPosition& p = packed[b];
        float v = 0.0f;

        if (plane < kPackHistory * kPackHistBlock)            // 0..103
        {
            const int slot = plane / kPackHistBlock;          // 0..7
            const int within = plane % kPackHistBlock;        // 0..12
            const bool slot_ok = (slot == 0)
                ? true : (((p.slot_valid >> slot) & 1u) != 0u);
            if (slot_ok)
            {
                if (within < kPackPiecePlanes)
                {
                    // Encoder lit (row=7-rank, col=file) for set bit at sq
                    // (rank=sq/8, file=sq%8) ⇒ source sq = (7-row)*8 + col.
                    const int sq = (kPackBoard - 1 - row) * kPackBoard + col;
                    const unsigned long long bb =
                        (unsigned long long)p.bitboards[slot][within];
                    v = ((bb >> sq) & 1ULL) ? 1.0f : 0.0f;
                }
                else // repetition plane (within == 12), full-plane 0/1
                {
                    v = ((p.slot_rep >> slot) & 1u) ? 1.0f : 0.0f;
                }
            }
        }
        else if (plane == kPackPlaneSTM)   v = p.stm ? 1.0f : 0.0f;
        else if (plane == kPackPlaneEP)
        {
            const int sq = (kPackBoard - 1 - row) * kPackBoard + col;
            v = (p.ep_square >= 0 && p.ep_square == sq) ? 1.0f : 0.0f;
        }
        else if (plane == kPackPlaneCWK)   v = (p.castling & 0x1u) ? 1.0f : 0.0f;
        else if (plane == kPackPlaneCWQ)   v = (p.castling & 0x2u) ? 1.0f : 0.0f;
        else if (plane == kPackPlaneCBK)   v = (p.castling & 0x4u) ? 1.0f : 0.0f;
        else if (plane == kPackPlaneCBQ)   v = (p.castling & 0x8u) ? 1.0f : 0.0f;
        else if (plane == kPackPlaneHMC)
        {
            const float hm = (float)p.halfmove / 100.0f;
            v = hm < 1.0f ? hm : 1.0f;
        }

        store(&out[idx], v);
    }
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

    // 3D grid: (batch, 64 cells); 128 threads = planes. Every element
    // written exactly once → no memset. batch ≤ 1024 ≪ 65535 grid limit.
    const dim3 grid(static_cast<unsigned>(batch),
                    static_cast<unsigned>(kPackBoard * kPackBoard), 1u);
    const dim3 block(128u, 1u, 1u);

    if (out_is_half)
        expand_kernel<__half><<<grid, block, 0, stream>>>(
            d_packed, static_cast<__half*>(d_out));
    else
        expand_kernel<float><<<grid, block, 0, stream>>>(
            d_packed, static_cast<float*>(d_out));
    return cudaGetLastError();
}

} // namespace chessrl::mcts::single
