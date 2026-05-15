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

template <typename T>
__global__ void expand_kernel(const SmallPackedPosition* __restrict__ packed,
                              T* __restrict__ out, long long total)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride)
    {
        // Decompose flat index → (b, row, col, plane). Layout:
        //   idx = b*7104 + (row*8 + col)*111 + plane
        const int plane = (int)(idx % kPackPlanes);
        long long t = idx / kPackPlanes;          // = b*64 + row*8 + col
        const int col = (int)(t % kPackBoard);
        t /= kPackBoard;
        const int row = (int)(t % kPackBoard);
        const int b = (int)(t / kPackBoard);

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

    // Every element is written exactly once → no memset needed.
    const long long total = (long long)batch * kPackElems;
    const int threads = 256;
    const int max_blocks = 65535;
    long long want = (total + threads - 1) / threads;
    const int blocks = (int)(want < max_blocks ? want : max_blocks);

    if (out_is_half)
        expand_kernel<__half><<<blocks, threads, 0, stream>>>(
            d_packed, static_cast<__half*>(d_out), total);
    else
        expand_kernel<float><<<blocks, threads, 0, stream>>>(
            d_packed, static_cast<float*>(d_out), total);
    return cudaGetLastError();
}

} // namespace chessrl::mcts::single
