// Fused forward pass of the encoder network (src/models/encoder.py at d_model 128, 8
// heads, d_ff 256, any depth up to kMaxLayers): one thread block per position runs the
// whole network, from the packed position to the policy logits and value outputs, in a
// single kernel launch.
//
// Work split: 8 warps, one per attention head. Warp w owns columns 16w..16w+15 of the
// residual stream for all 64 tokens and keeps them in registers, in the mma.sync
// accumulator layout, so the out and down projections accumulate straight into it.
// Normalized activations go through shared memory as FP16 (the A operand of every
// matmul). Weights stream from L2 into registers as pre-packed B fragments
// (src/export/to_cnet.py); every weight is read by exactly one warp per block, so there
// is nothing to share through shared memory. Accumulation is FP32, norms are FP32.
// Attention for head w stays in warp w's registers: Q and K come out of the qkv matmul
// already in the operand layouts of the score matmul.
//
// Fragment layout (m16n8k16): lane = 4g + t. An accumulator tile acc[m][n][r] holds row
// 16m + g + 8(r / 2), column 8·tile(n) + 2t + r % 2.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

#include "net/encoder_params.hpp"

namespace chessrl::net
{
namespace
{

using mcts::single::SmallPackedPosition;
using namespace mcts::single;  // plane-layout constants

constexpr int kTok = 64;
constexpr int kD = 128;
constexpr int kHd = 16;
constexpr int kFf = 256;
constexpr int kWarps = kD / kHd;
constexpr int kThreads = 32 * kWarps;
constexpr int kLdH = kD + 8;   // FP16 row strides, padded so the 8 rows of an ldmatrix
constexpr int kLdF = kFf + 8;  // land on distinct banks
constexpr int kLdV = kTok + 8;
constexpr int kLdS = kTok + 4;  // FP32 policy score rows
constexpr float kEps = 1e-6f;

struct alignas(16) Smem
{
    __half h[kTok * kLdH];          // input planes, then each norm's output, then the final h
    __half o[kTok * kLdH];          // attention output, all heads
    __half f[kTok * kLdF];          // feed-forward hidden; policy q | k; policy scores (FP32)
    __half v[kWarps * kHd * kLdV];  // per head: V transposed, [dim][token]
    float red[kWarps * kTok];       // per-warp partial sums of squares, per token
    float pooled[kD];
    float hidden[4 * kD];
    float heads[16];
    unsigned long long packed[(sizeof(SmallPackedPosition) + 7) / 8];
};
static_assert(sizeof(Smem) <= 99 * 1024, "shared memory per block");
static_assert(sizeof(Smem::f) >= (kTok + 4) * kLdS * sizeof(float), "policy scores fit in f");

__device__ __forceinline__ uint32_t h2(float a, float b)
{
    const __half2 v = __floats2half2_rn(a, b);
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    return u;
}

__device__ __forceinline__ float2 unh2(uint32_t u)
{
    __half2 v;
    memcpy(&v, &u, sizeof(v));
    return __half22float2(v);
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const __half* p)
{
    const auto a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

// c += a (16x16, row major) * b (16x8, column major); FP16 inputs, FP32 accumulators.
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The same with FP16 accumulators: c[0] holds row g, c[1] row g + 8 (two halves each). The
// tensor cores of GeForce cards run this at twice the FP32-accumulator rate.
__device__ __forceinline__ void mma_h(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, "
        "{%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float exp2_approx(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float silu(float x)
{
    return x / (1.f + __expf(-x));
}

template <int NT>
__device__ __forceinline__ void zero(float (&a)[4][NT][4])
{
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int r = 0; r < 4; ++r) a[m][n][r] = 0.f;
}

// acc[m][n] += A[rows of tile m] · W[n-tile tile[n]]ᵀ over K = 32·KP, for the 64 tokens.
// A: FP16 in shared memory, row stride lda. W: B fragments packed by to_cnet.pack_b,
// one 16-byte load per lane per (n-tile, 32 k); the next k-pair is fetched while the
// current one is multiplied. kChunk = 0: FP32 accumulators. kChunk > 0: FP16 accumulators,
// added into acc (FP32) every kChunk k-pairs (32·kChunk terms) and at the end.
template <int KP, int NT, int kChunk>
__device__ __forceinline__ void warp_gemm(float (&acc)[4][NT][4], const __half* a, int lda,
                                          const void* weights, const int (&tile)[NT], int lane)
{
    const uint4* w = static_cast<const uint4*>(weights);
    const __half* arow = a + (lane & 15) * lda + (lane >> 4) * 8;
    uint4 b[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) b[n] = __ldg(w + tile[n] * KP * 32 + lane);
    uint32_t h[4][NT][2];
    if constexpr (kChunk > 0)
    {
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n) h[m][n][0] = h[m][n][1] = 0u;
    }
#pragma unroll
    for (int p = 0; p < KP; ++p)
    {
        uint4 next[NT];
        if (p + 1 < KP)
        {
#pragma unroll
            for (int n = 0; n < NT; ++n) next[n] = __ldg(w + (tile[n] * KP + p + 1) * 32 + lane);
        }
#pragma unroll
        for (int s = 0; s < 2; ++s)
        {
#pragma unroll
            for (int m = 0; m < 4; ++m)
            {
                uint32_t af[4];
                ldsm4(af, arow + m * 16 * lda + p * 32 + s * 16);
#pragma unroll
                for (int n = 0; n < NT; ++n)
                {
                    if constexpr (kChunk == 0)
                        mma(acc[m][n], af, s ? b[n].z : b[n].x, s ? b[n].w : b[n].y);
                    else
                        mma_h(h[m][n], af, s ? b[n].z : b[n].x, s ? b[n].w : b[n].y);
                }
            }
        }
        if constexpr (kChunk > 0)
        {
            if ((p + 1) % kChunk == 0 || p + 1 == KP)
            {
#pragma unroll
                for (int m = 0; m < 4; ++m)
#pragma unroll
                    for (int n = 0; n < NT; ++n)
                    {
                        const float2 lo = unh2(h[m][n][0]), hi = unh2(h[m][n][1]);
                        acc[m][n][0] += lo.x;
                        acc[m][n][1] += lo.y;
                        acc[m][n][2] += hi.x;
                        acc[m][n][3] += hi.y;
                        h[m][n][0] = h[m][n][1] = 0u;
                    }
            }
        }
        if (p + 1 < KP)
        {
#pragma unroll
            for (int n = 0; n < NT; ++n) b[n] = next[n];
        }
    }
}

// 1 / RMS of each of this thread's 8 rows (rstd[m][i]: row 16m + g + 8i), over all 128
// columns: per-warp partial sums through shared memory, one barrier.
__device__ __forceinline__ void row_rstd(const float (&x)[4][2][4], float* red, int warp, int lane,
                                         float (&rstd)[4][2])
{
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            float s = 0.f;
#pragma unroll
            for (int n = 0; n < 2; ++n)
                s += x[m][n][2 * i] * x[m][n][2 * i] + x[m][n][2 * i + 1] * x[m][n][2 * i + 1];
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            if (t == 0) red[warp * kTok + 16 * m + g + 8 * i] = s;
        }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const int row = 16 * m + g + 8 * i;
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) s += red[w * kTok + row];
            rstd[m][i] = rsqrtf(s * (1.f / kD) + kEps);
        }
}

// dst = RMSNorm(x) · weight, FP16, then a barrier so dst is ready as an A operand.
__device__ __forceinline__ void norm_to_smem(const float (&x)[4][2][4], const float* __restrict__ weight,
                                             __half* dst, float* red, int warp, int lane)
{
    const int g = lane >> 2, t = lane & 3;
    float rstd[4][2];
    row_rstd(x, red, warp, lane, rstd);
#pragma unroll
    for (int n = 0; n < 2; ++n)
    {
        const int col = 16 * warp + 8 * n + 2 * t;
        const float2 wt = __ldg(reinterpret_cast<const float2*>(weight + col));
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                const int row = 16 * m + g + 8 * i;
                *reinterpret_cast<uint32_t*>(dst + row * kLdH + col) =
                    h2(x[m][n][2 * i] * rstd[m][i] * wt.x, x[m][n][2 * i + 1] * rstd[m][i] * wt.y);
            }
    }
    __syncthreads();
}

// RMSNorm over one head's 16 dims (accumulator tiles n0, n0 + 1), times the norm weight.
template <int NT>
__device__ __forceinline__ void head_norm(float (&a)[4][NT][4], int n0, const float* __restrict__ weight,
                                          int t)
{
    const float2 w0 = __ldg(reinterpret_cast<const float2*>(weight + 2 * t));
    const float2 w1 = __ldg(reinterpret_cast<const float2*>(weight + 8 + 2 * t));
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            float* lo = &a[m][n0][2 * i];
            float* hi = &a[m][n0 + 1][2 * i];
            float s = lo[0] * lo[0] + lo[1] * lo[1] + hi[0] * hi[0] + hi[1] * hi[1];
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            const float r = rsqrtf(s * (1.f / kHd) + kEps);
            lo[0] *= r * w0.x;
            lo[1] *= r * w0.y;
            hi[0] *= r * w1.x;
            hi[1] *= r * w1.y;
        }
}

// Attention for head `warp`. qk tiles 0-1 hold Q (already scaled by log2(e) / 4 through
// the q norm weight), tiles 2-3 hold K; vt holds V transposed. Writes the head's 16
// output columns of `o`.
__device__ __forceinline__ void attention(const float (&qk)[4][4][4], const __half* vt,
                                          const void* pair_bias, __half* o, int warp, int lane)
{
    const int g = lane >> 2, t = lane & 3;
    // K as B operand: key tile j (keys 8j..8j+7) is row half j % 2 of K's row tile j / 2.
    uint32_t kb[8][2];
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        const int m = j >> 1, i = j & 1;
        kb[j][0] = h2(qk[m][2][2 * i], qk[m][2][2 * i + 1]);
        kb[j][1] = h2(qk[m][3][2 * i], qk[m][3][2 * i + 1]);
    }
    // V as B operand: k-step s covers keys 16s..16s+15, n-tile d covers dims 8d..8d+7.
    uint32_t vb[4][2][2];
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
        for (int d = 0; d < 2; ++d)
        {
            const __half* p = vt + (8 * d + g) * kLdV + 16 * s + 2 * t;
            vb[s][d][0] = *reinterpret_cast<const uint32_t*>(p);
            vb[s][d][1] = *reinterpret_cast<const uint32_t*>(p + 8);
        }
    const uint4* bias = static_cast<const uint4*>(pair_bias) + warp * 4 * 4 * 32 + lane;

#pragma unroll
    for (int m = 0; m < 4; ++m)
    {
        const uint32_t qa[4] = {h2(qk[m][0][0], qk[m][0][1]), h2(qk[m][0][2], qk[m][0][3]),
                                h2(qk[m][1][0], qk[m][1][1]), h2(qk[m][1][2], qk[m][1][3])};
        // Scores start from the pair bias (packed by to_cnet.pack_pair_bias, × log2(e)).
        float sc[8][4];
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            const uint4 bv = __ldg(bias + (m * 4 + c) * 32);
            const float2 a0 = unh2(bv.x), a1 = unh2(bv.y), a2 = unh2(bv.z), a3 = unh2(bv.w);
            sc[2 * c][0] = a0.x;
            sc[2 * c][1] = a0.y;
            sc[2 * c][2] = a1.x;
            sc[2 * c][3] = a1.y;
            sc[2 * c + 1][0] = a2.x;
            sc[2 * c + 1][1] = a2.y;
            sc[2 * c + 1][2] = a3.x;
            sc[2 * c + 1][3] = a3.y;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) mma(sc[j], qa, kb[j][0], kb[j][1]);

        // Softmax over the 64 keys (base 2) of rows g (r = 0, 1) and g + 8 (r = 2, 3).
        float mx0 = sc[0][0], mx1 = sc[0][2];
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            mx0 = fmaxf(mx0, fmaxf(sc[j][0], sc[j][1]));
            mx1 = fmaxf(mx1, fmaxf(sc[j][2], sc[j][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        float l0 = 0.f, l1 = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            sc[j][0] = exp2_approx(sc[j][0] - mx0);
            sc[j][1] = exp2_approx(sc[j][1] - mx0);
            sc[j][2] = exp2_approx(sc[j][2] - mx1);
            sc[j][3] = exp2_approx(sc[j][3] - mx1);
            l0 += sc[j][0] + sc[j][1];
            l1 += sc[j][2] + sc[j][3];
        }
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);

        // P · V: the probabilities' accumulator tiles are already the A operand.
        float acc[2][4] = {};
#pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            const uint32_t pa[4] = {h2(sc[2 * s][0], sc[2 * s][1]), h2(sc[2 * s][2], sc[2 * s][3]),
                                    h2(sc[2 * s + 1][0], sc[2 * s + 1][1]),
                                    h2(sc[2 * s + 1][2], sc[2 * s + 1][3])};
#pragma unroll
            for (int d = 0; d < 2; ++d) mma(acc[d], pa, vb[s][d][0], vb[s][d][1]);
        }
        const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
        for (int d = 0; d < 2; ++d)
        {
            const int col = 16 * warp + 8 * d + 2 * t;
            *reinterpret_cast<uint32_t*>(o + (16 * m + g) * kLdH + col) = h2(acc[d][0] * i0, acc[d][1] * i0);
            *reinterpret_cast<uint32_t*>(o + (16 * m + g + 8) * kLdH + col) =
                h2(acc[d][2] * i1, acc[d][3] * i1);
        }
    }
}

// The input plane `plane` of token `cell` (row-major 8x8, row 0 = rank 8), as
// expand_packed_to_dense (small_encoding.cpp) builds it; planes 111..127 are the zero
// padding of the embedding's k loop.
__device__ __forceinline__ float plane_value(const SmallPackedPosition& p, int cell, int plane)
{
    const int sq = (kPackBoard - 1 - (cell >> 3)) * kPackBoard + (cell & 7);
    if (plane < kPackPlaneSTM)
    {
        const int slot = plane / kPackHistBlock, within = plane % kPackHistBlock;
        if (slot != 0 && !((p.slot_valid >> slot) & 1u)) return 0.f;
        if (within < kPackPiecePlanes) return ((p.bitboards[slot][within] >> sq) & 1ull) ? 1.f : 0.f;
        return ((p.slot_rep >> slot) & 1u) ? 1.f : 0.f;
    }
    switch (plane)
    {
    case kPackPlaneSTM: return p.stm ? 1.f : 0.f;
    case kPackPlaneEP: return p.ep_square == sq ? 1.f : 0.f;
    case kPackPlaneCWK: return (p.castling & 0x1u) ? 1.f : 0.f;
    case kPackPlaneCWQ: return (p.castling & 0x2u) ? 1.f : 0.f;
    case kPackPlaneCBK: return (p.castling & 0x4u) ? 1.f : 0.f;
    case kPackPlaneCBQ: return (p.castling & 0x8u) ? 1.f : 0.f;
    case kPackPlaneHMC: return fminf(static_cast<float>(p.halfmove) / 100.f, 1.f);
    default: return 0.f;
    }
}

template <int kChunk>
__global__ void __launch_bounds__(kThreads, 1)
    encoder_forward(__grid_constant__ const EncoderParams P, const SmallPackedPosition* __restrict__ packed,
                    const float* __restrict__ dense, unsigned char* __restrict__ out)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& s = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int b = blockIdx.x;

    // ---- Input planes into s.h as [token][plane], FP16.
    if (packed)
    {
        const auto* src = reinterpret_cast<const unsigned long long*>(packed + b);
        for (int i = tid; i < static_cast<int>(sizeof(s.packed) / 8); i += kThreads) s.packed[i] = src[i];
        __syncthreads();
        const auto& pos = *reinterpret_cast<const SmallPackedPosition*>(s.packed);
        for (int i = tid; i < kTok * 128; i += kThreads)
            s.h[(i >> 7) * kLdH + (i & 127)] = __float2half(plane_value(pos, i >> 7, i & 127));
    }
    else
    {
        const float* src = dense + static_cast<size_t>(b) * kTok * kPackPlanes;
        for (int i = tid; i < kTok * 128; i += kThreads)
        {
            const int tok = i >> 7, pl = i & 127;
            s.h[tok * kLdH + pl] = __float2half(pl < kPackPlanes ? src[tok * kPackPlanes + pl] : 0.f);
        }
    }
    __syncthreads();

    // ---- Embedding: x0 = planes · We + be + pos. x (FP32) and x0 (FP16) stay in registers.
    const int cols[2] = {2 * warp, 2 * warp + 1};
    float x[4][2][4];
    zero(x);
    warp_gemm<4, 2, kChunk>(x, s.h, kLdH, P.embed_w, cols, lane);
    uint32_t x0[4][2][2];
#pragma unroll
    for (int n = 0; n < 2; ++n)
    {
        const int col = 16 * warp + 8 * n + 2 * t;
        const float2 eb = __ldg(reinterpret_cast<const float2*>(P.embed_b + col));
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                const int row = 16 * m + g + 8 * i;
                const float2 pe = __ldg(reinterpret_cast<const float2*>(P.pos + row * kD + col));
                x[m][n][2 * i] += eb.x + pe.x;
                x[m][n][2 * i + 1] += eb.y + pe.y;
                x0[m][n][i] = h2(x[m][n][2 * i], x[m][n][2 * i + 1]);
            }
    }

    for (int l = 0; l < P.layers; ++l)
    {
        const LayerParams& L = P.layer[l];
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    const float2 v = unh2(x0[m][n][i]);
                    x[m][n][2 * i] += L.gain * v.x;
                    x[m][n][2 * i + 1] += L.gain * v.y;
                }
        norm_to_smem(x, L.norm1, s.h, s.red, warp, lane);

        // ---- Attention, head = warp. V first, transposed into the warp's own slice.
        __half* vt = s.v + warp * kHd * kLdV;
        {
            float v[4][2][4];
            zero(v);
            const int vtile[2] = {32 + 2 * warp, 33 + 2 * warp};
            warp_gemm<4, 2, kChunk>(v, s.h, kLdH, L.qkv, vtile, lane);
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                        vt[(8 * n + 2 * t + (r & 1)) * kLdV + 16 * m + g + 8 * (r >> 1)] =
                            __float2half(v[m][n][r]);
        }
        {
            float qk[4][4][4];
            zero(qk);
            const int qktile[4] = {2 * warp, 2 * warp + 1, 16 + 2 * warp, 17 + 2 * warp};
            warp_gemm<4, 4, kChunk>(qk, s.h, kLdH, L.qkv, qktile, lane);
            head_norm(qk, 0, L.qnorm, t);
            head_norm(qk, 2, L.knorm, t);
            __syncwarp();
            attention(qk, vt, P.pair_bias, s.o, warp, lane);
        }
        __syncthreads();
        warp_gemm<4, 2, kChunk>(x, s.o, kLdH, L.out, cols, lane);

        // ---- SwiGLU feed-forward, in two passes of 16 gate + 16 up columns per warp.
        norm_to_smem(x, L.norm2, s.h, s.red, warp, lane);
#pragma unroll
        for (int pass = 0; pass < 2; ++pass)
        {
            float u[4][4][4];
            zero(u);
            const int c0 = 4 * warp + 2 * pass;
            const int utile[4] = {c0, c0 + 1, 32 + c0, 33 + c0};
            warp_gemm<4, 4, kChunk>(u, s.h, kLdH, L.up, utile, lane);
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                const int col = 8 * (c0 + n) + 2 * t;
#pragma unroll
                for (int m = 0; m < 4; ++m)
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                    {
                        const int row = 16 * m + g + 8 * i;
                        *reinterpret_cast<uint32_t*>(s.f + row * kLdF + col) =
                            h2(silu(u[m][n][2 * i]) * u[m][n + 2][2 * i],
                               silu(u[m][n][2 * i + 1]) * u[m][n + 2][2 * i + 1]);
                    }
            }
        }
        __syncthreads();
        warp_gemm<8, 2, kChunk>(x, s.f, kLdF, L.down, cols, lane);
    }

    // ---- Final norm into s.h, and the token mean of the normed stream (value input).
    {
        float rstd[4][2];
        row_rstd(x, s.red, warp, lane, rstd);
#pragma unroll
        for (int n = 0; n < 2; ++n)
        {
            const int col = 16 * warp + 8 * n + 2 * t;
            const float2 wt = __ldg(reinterpret_cast<const float2*>(P.norm + col));
            float c0 = 0.f, c1 = 0.f;
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    const int row = 16 * m + g + 8 * i;
                    const float y0 = x[m][n][2 * i] * rstd[m][i] * wt.x;
                    const float y1 = x[m][n][2 * i + 1] * rstd[m][i] * wt.y;
                    *reinterpret_cast<uint32_t*>(s.h + row * kLdH + col) = h2(y0, y1);
                    c0 += y0;
                    c1 += y1;
                }
#pragma unroll
            for (int off = 4; off < 32; off <<= 1)
            {
                c0 += __shfl_xor_sync(0xffffffffu, c0, off);
                c1 += __shfl_xor_sync(0xffffffffu, c1, off);
            }
            if (g == 0)
            {
                s.pooled[col] = c0 * (1.f / kTok);
                s.pooled[col + 1] = c1 * (1.f / kTok);
            }
        }
    }
    __syncthreads();

    // ---- Policy: q | k projections into s.f (q pre-scaled by 1/sqrt(d) in the file).
    {
        float a[4][4][4];
        zero(a);
        const int ptile[4] = {2 * warp, 2 * warp + 1, 16 + 2 * warp, 17 + 2 * warp};
        warp_gemm<4, 4, kChunk>(a, s.h, kLdH, P.pol_w, ptile, lane);
#pragma unroll
        for (int n = 0; n < 4; ++n)
        {
            const int col = 8 * ptile[n] + 2 * t;
            const float2 bb = __ldg(reinterpret_cast<const float2*>(P.pol_b + col));
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    const int row = 16 * m + g + 8 * i;
                    *reinterpret_cast<uint32_t*>(s.f + row * kLdF + col) =
                        h2(a[m][n][2 * i] + bb.x, a[m][n][2 * i + 1] + bb.y);
                }
        }
    }
    __syncthreads();

    // Scores for to-tokens 8·warp..8·warp+7: ft[from][to] = q_from · k_to and
    // pt[promotion][to] = promo · k_to (promotion rows 0-3 of one extra row tile).
    float ft[4][4] = {}, pt[4] = {};
    {
        const auto* promo = static_cast<const uint32_t*>(P.promo);  // [4][64] FP16 pairs
        const __half* arow = s.f + (lane & 15) * kLdF + (lane >> 4) * 8;
        const __half* krow = s.f + (8 * warp + g) * kLdF + kD + 2 * t;
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks)
        {
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow + 16 * ks);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + 16 * ks + 8);
#pragma unroll
            for (int m = 0; m < 4; ++m)
            {
                uint32_t af[4];
                ldsm4(af, arow + 16 * m * kLdF + 16 * ks);
                mma(ft[m], af, b0, b1);
            }
            const uint32_t pa[4] = {g < 4 ? __ldg(promo + g * 64 + 8 * ks + t) : 0u, 0u,
                                    g < 4 ? __ldg(promo + g * 64 + 8 * ks + 4 + t) : 0u, 0u};
            mma(pt, pa, b0, b1);
        }
    }
    __syncthreads();
    float* sc = reinterpret_cast<float*>(s.f);  // [68][kLdS]: from-token rows, then promotions
    {
        const int col = 8 * warp + 2 * t;
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                *reinterpret_cast<float2*>(sc + (16 * m + g + 8 * i) * kLdS + col) =
                    make_float2(ft[m][2 * i], ft[m][2 * i + 1]);
        if (g < 4) *reinterpret_cast<float2*>(sc + (kTok + g) * kLdS + col) = make_float2(pt[0], pt[1]);
    }
    __syncthreads();
    auto* policy = reinterpret_cast<__half*>(out + static_cast<size_t>(b) * kRecordBytes);
    for (int i = tid; i < kPolicySize; i += kThreads)
    {
        const unsigned code = __ldg(P.vocab + i);
        const int to = (code >> 6) & 63;
        const float v = (code >> 15)
            ? P.special
            : sc[(code & 63) * kLdS + to] + sc[(kTok + ((code >> 12) & 3)) * kLdS + to];
        policy[i] = __float2half(v);
    }

    // ---- Value heads: MLPs on the pooled vector (first layers stored [mlp][in][out]).
    const int n_mlp = P.n_wdl + 1 + P.mlhq_from_q;
    for (int u = tid; u < n_mlp * kD; u += kThreads)
    {
        const __half* w = static_cast<const __half*>(P.mlp1_w) + (u >> 7) * kD * kD + (u & 127);
        float acc = __ldg(P.mlp1_b + u);
#pragma unroll 8
        for (int k = 0; k < kD; ++k) acc += __half2float(__ldg(w + k * kD)) * s.pooled[k];
        s.hidden[u] = silu(acc);
    }
    __syncthreads();
    const int n_out = 3 * P.n_wdl + 1 + P.mlhq_from_q;
    for (int o = warp; o < n_out; o += kWarps)
    {
        const int mlp = o < 3 * P.n_wdl ? o / 3 : P.n_wdl + (o - 3 * P.n_wdl);
        float acc = 0.f;
        for (int k = lane; k < kD; k += 32) acc += __ldg(P.mlp2_w + o * kD + k) * s.hidden[mlp * kD + k];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) s.heads[o] = acc + __ldg(P.mlp2_b + o);
    }
    __syncthreads();
    if (tid == 0)
    {
        float wdl[3] = {0.f, 0.f, 0.f};
        for (int k = 0; k < P.n_wdl; ++k)
        {
            const float* z = s.heads + 3 * k;
            const float mx = fmaxf(z[0], fmaxf(z[1], z[2]));
            const float e0 = expf(z[0] - mx), e1 = expf(z[1] - mx), e2 = expf(z[2] - mx);
            const float scale = 1.f / ((e0 + e1 + e2) * P.n_wdl);
            wdl[0] += e0 * scale;
            wdl[1] += e1 * scale;
            wdl[2] += e2 * scale;
        }
        auto* aux = reinterpret_cast<float*>(out + static_cast<size_t>(b) * kRecordBytes + kAuxOffset);
        aux[0] = wdl[0];
        aux[1] = wdl[1];
        aux[2] = wdl[2];
        aux[3] = s.heads[3 * P.n_wdl];
        aux[4] = P.mlhq_from_q ? s.heads[3 * P.n_wdl + 1] : aux[3];
    }
}

template <int kChunk>
cudaError_t launch(const EncoderParams& params, const SmallPackedPosition* packed, const float* dense,
                   int batch, void* out, cudaStream_t stream)
{
    static const cudaError_t setup = cudaFuncSetAttribute(
        encoder_forward<kChunk>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sizeof(Smem)));
    if (setup != cudaSuccess) return setup;
    encoder_forward<kChunk><<<batch, kThreads, sizeof(Smem), stream>>>(params, packed, dense,
                                                                      static_cast<unsigned char*>(out));
    return cudaGetLastError();
}

// CHESSRL_FP16_ACC: 0 (default) keeps FP32 accumulators; N > 0 sums N terms (32, 64, or 256 for
// whole sums) in FP16 between FP32 additions, in the projection matrix multiplies. The score
// multiplies (attention, policy logits) and the residual stream stay FP32.
int fp16_chunk()
{
    static const int chunk = [] {
        const char* env = std::getenv("CHESSRL_FP16_ACC");
        const int terms = env ? std::atoi(env) : 0;
        if (terms <= 0) return 0;
        if (terms <= 32) return 1;
        if (terms <= 64) return 2;
        return 8;
    }();
    return chunk;
}

} // namespace

cudaError_t launch_encoder_forward(const EncoderParams& params,
                                   const SmallPackedPosition* packed,
                                   const float* dense,
                                   int batch,
                                   void* out,
                                   cudaStream_t stream)
{
    if (batch <= 0) return cudaSuccess;
    switch (fp16_chunk())
    {
    case 1: return launch<1>(params, packed, dense, batch, out, stream);
    case 2: return launch<2>(params, packed, dense, batch, out, stream);
    case 8: return launch<8>(params, packed, dense, batch, out, stream);
    default: return launch<0>(params, packed, dense, batch, out, stream);
    }
}

} // namespace chessrl::net
