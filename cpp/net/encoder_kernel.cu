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
// INT8 networks (to_cnet --int8, src/export/int8.py) run the projections as m16n8k32 INT8 multiplies with INT32
// sums: weights with per-row scales, activations quantized per token where they are produced (in the norms, or a
// pass over the attention and SwiGLU outputs in shared memory). The rest is the FP16 kernel's arithmetic.
//
// Fragment layout (m16n8k16): lane = 4g + t. An accumulator tile acc[m][n][r] holds row
// 16m + g + 8(r / 2), column 8·tile(n) + 2t + r % 2.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "net/encoder_params.hpp"
#include "net/planes.cuh"

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
constexpr int kLdV = kTok + 8; // land on distinct banks
constexpr int kLdS = kTok + 4;  // FP32 policy score rows
constexpr float kEps = 1e-6f;
// The norms' per-warp partials, [row][warp] so a row's come in 16-byte loads, rows padded so the 8 rows a warp
// reads at once land on distinct banks: FP16 networks keep sums of squares (kRedLd floats per row), int8 ones
// (sum of squares, max |x · w|) pairs (kRed2Ld float2 per row).
constexpr int kRedLd = 12;
static_assert(kRedLd > 8, "a padding slot per row for the norms' 1/RMS");
constexpr int kRed2Ld = 10;
static_assert(kRed2Ld > 8, "a padding slot per row for the int8 norms' row totals");
// INT8 projections: n-tiles per pass of warp_gemm_i8.
constexpr int kI8Group = 2;
// Attention multiplies with FP16 accumulators (the full-rate ones on GeForce cards): bit 1 P · V (probabilities
// at most 1), bit 2 the scores (a single k16 step from the FP16 pair bias). Softmax and normalizer stay FP32.
constexpr int kAttnHalf = 3;

// The instrumented build's phases (CHESSRL_KERNEL_TIMING=1): cycles thread 0 spends from one mark to the next,
// so a phase ending in a barrier includes the wait for the other warps. Per-layer phases sum over the layers.
constexpr int kPhases = 15;
const char* const kPhaseNames[kPhases] = {
    "input planes", "embedding", "norm1", "v projection", "q k projection + head norms", "attention",
    "out projection", "norm2", "ffn up + swiglu", "ffn down", "final norm + pooling", "policy q k projection",
    "policy scores", "policy output", "value heads"};
__device__ unsigned long long g_phase[kPhases];

// 41 KB, so that two blocks fit on an SM. Two FP16 buffers do all the work. h: the input planes,
// each norm's output, the attention output (written once every warp has its q, k and v) and the
// policy's q. v: V transposed (per head, [dim][token]), then one half of the feed-forward hidden
// layer at a time, then the policy's k, both [token][kLdH]. The FP32 policy scores span both.
// An int8 network quantizes each matmul's A rows of h or v in place (quantize_rows), as int8 rows at the same
// 272-byte stride (the ldmatrix rows stay on distinct banks).
struct alignas(16) Smem
{
    __half h[kTok * kLdH];
    __half v[kWarps * kHd * kLdV];
    alignas(16) float red[2 * kRed2Ld * kTok];  // the norms' per-warp partials (kRedLd, kRed2Ld)
    float pooled[kD];
    float hidden[4 * kD];
    float heads[16];
    float scale_h[kTok];  // int8 networks: the token scales of h's and v's int8 rows (quantize_rows)
    float scale_v[kTok];
    unsigned long long packed[(sizeof(SmallPackedPosition) + 7) / 8];
};
static_assert(sizeof(Smem) <= 48 * 1024, "two blocks per SM");
static_assert(sizeof(Smem::v) >= kTok * kLdH * sizeof(__half), "a hidden half and the policy's k fit in v");
static_assert(offsetof(Smem, v) == sizeof(Smem::h), "v follows h");
static_assert(sizeof(Smem::h) + sizeof(Smem::v) >= (kTok + 4) * kLdS * sizeof(float), "policy scores fit in h and v");

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

// c += a (16x32, row major) * b (32x8, column major); INT8 inputs, INT32 accumulators (int8 networks). The A
// fragment has the byte layout of m16n8k16's FP16 one (rows g, g + 8; bytes 4t.., 16 + 4t..), so ldsm4 loads it
// from 16 rows of 32 bytes; c has the FP32 accumulator layout.
__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const int8_t* p)
{
    ldsm4(r, reinterpret_cast<const __half*>(p));
}

// int8 <-> float without the conversion unit (F2I / I2F: 16 per clock per SM on sm_86 / sm_89, a quarter of the
// FP32 rate, and the INT8 path converts every accumulator and every quantized activation). Adding 1.5 · 2^23 rounds
// to the nearest integer (ties to even, as __float2int_rn) and leaves it in the low mantissa bits; the same offset
// turns an int32 into a float. Exact for |v|, |c| < 2^22 (sums of 256 int8 products stay below 2^22).
__device__ __forceinline__ uint32_t round_to_byte(float v)
{
    return __float_as_uint(v + 12582912.f) & 0xFFu;
}

__device__ __forceinline__ float int_to_float(int c)
{
    return __int_as_float(c + 0x4B400000) - 12582912.f;
}

__device__ __forceinline__ float exp2_approx(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// The fast division (reciprocal and multiply, 2 ulp; 0 once e^-x overflows, silu's limit): the IEEE one's Newton
// steps and slow-path check were a large share of the feed-forward's instructions, 64 calls per thread per layer.
__device__ __forceinline__ float silu(float x)
{
    return __fdividef(x, 1.f + __expf(-x));
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
// added into acc (FP32) every kChunk k-pairs (32·kChunk terms) and at the end. KS, p0: the
// weights hold KS k-pairs per n-tile and this call multiplies k-pairs p0..p0 + KP - 1 of them
// (A holding just those KP·32 columns).
// kMock (timing experiments only, wrong results; CHESSRL_KERNEL_MOCK): bit 1 loads the weights of every other
// n-tile only (half the weight bytes), bit 2 skips the second k16 multiply of each k-pair (half the multiplies),
// bit 4 reuses the first k16 A fragment for the second (half the A loads). 7 = what INT8 would do to this loop.
template <int KP, int NT, int kChunk, int KS = KP, int kMock = 0>
__device__ __forceinline__ void warp_gemm(float (&acc)[4][NT][4], const __half* a, int lda,
                                          const void* weights, const int (&tile)[NT], int lane, int p0 = 0)
{
    const uint4* w = static_cast<const uint4*>(weights);
    const __half* arow = a + (lane & 15) * lda + (lane >> 4) * 8;
    uint4 b[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n)
        b[n] = ((kMock & 1) && (n & 1)) ? b[n - 1] : __ldg(w + (tile[n] * KS + p0) * 32 + lane);
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
            for (int n = 0; n < NT; ++n)
                next[n] = ((kMock & 1) && (n & 1)) ? next[n - 1] : __ldg(w + (tile[n] * KS + p0 + p + 1) * 32 + lane);
        }
        uint32_t af0[4][4];  // mock bit 4: the s = 0 fragments, reused for s = 1
#pragma unroll
        for (int s = 0; s < 2; ++s)
        {
            if ((kMock & 2) && s == 1) continue;
#pragma unroll
            for (int m = 0; m < 4; ++m)
            {
                uint32_t af[4];
                if ((kMock & 4) && s == 1)
                {
#pragma unroll
                    for (int r = 0; r < 4; ++r) af[r] = af0[m][r];
                }
                else
                {
                    ldsm4(af, arow + m * 16 * lda + p * 32 + s * 16);
                    if constexpr ((kMock & 4) != 0)
                    {
#pragma unroll
                        for (int r = 0; r < 4; ++r) af0[m][r] = af[r];
                    }
                }
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

// The INT8 projections (int8 networks): acc[m][n] = (kAdd: +=) Σ over K = 64·KP of A[row][k] · W[n][k], summed in
// INT32 and then times sa[row] · sw[n] (the token's scale, from quantize_rows, and the weight row's). A: int8 rows
// of stride lda bytes in shared memory. W: fragments packed by to_cnet.pack_b8, one 16-byte load per lane per
// (n-tile, 64 k), the next one fetched while the current one is multiplied. KS, p0: as in warp_gemm, counted in
// 64-k chunks.
template <int KP, int NT, bool kAdd, int KS = KP>
__device__ __forceinline__ void warp_gemm_i8(float (&acc)[4][NT][4], const int8_t* a, int lda, const void* weights,
                                             const float* __restrict__ sw, const float* sa, const int (&tile)[NT],
                                             int lane, int p0 = 0)
{
    // Registers, under the 128 of two blocks per SM next to the residual stream: n-tiles kI8Group at a time, all
    // their KP fragments loaded at once (one wait on L2 per group), then the INT32 sums of two m-tiles at a time
    // (A is read from shared memory once per n-group either way).
    constexpr int NG = NT < kI8Group ? NT : kI8Group;
    const uint4* w = static_cast<const uint4*>(weights);
    const int8_t* arow = a + (lane & 15) * lda + (lane >> 4) * 16;
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int n0 = 0; n0 < NT; n0 += NG)
    {
        uint4 b[KP][NG];
#pragma unroll
        for (int p = 0; p < KP; ++p)
#pragma unroll
            for (int n = 0; n < NG; ++n) b[p][n] = __ldg(w + (tile[n0 + n] * KS + p0 + p) * 32 + lane);
        float2 ws[NG];
#pragma unroll
        for (int n = 0; n < NG; ++n) ws[n] = __ldg(reinterpret_cast<const float2*>(sw + 8 * tile[n0 + n] + 2 * t));
#pragma unroll
        for (int mh = 0; mh < 2; ++mh)
        {
            int c[2][NG][4];
#pragma unroll
            for (int mm = 0; mm < 2; ++mm)
#pragma unroll
                for (int n = 0; n < NG; ++n)
#pragma unroll
                    for (int r = 0; r < 4; ++r) c[mm][n][r] = 0;
#pragma unroll
            for (int p = 0; p < KP; ++p)
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int mm = 0; mm < 2; ++mm)
                    {
                        uint32_t af[4];
                        ldsm4(af, arow + (2 * mh + mm) * 16 * lda + p * 64 + s * 32);
#pragma unroll
                        for (int n = 0; n < NG; ++n)
                            mma_s8(c[mm][n], af, s ? b[p][n].z : b[p][n].x, s ? b[p][n].w : b[p][n].y);
                    }
#pragma unroll
            for (int mm = 0; mm < 2; ++mm)
            {
                const int m = 2 * mh + mm;
                const float ra0 = sa[16 * m + g], ra1 = sa[16 * m + g + 8];
#pragma unroll
                for (int n = 0; n < NG; ++n)
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                    {
                        const float v = int_to_float(c[mm][n][r]) * ((r >> 1 ? ra1 : ra0) * ((r & 1) ? ws[n].y : ws[n].x));
                        if constexpr (kAdd)
                            acc[m][n0 + n][r] += v;
                        else
                            acc[m][n0 + n][r] = v;
                    }
            }
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
            if (t == 0) red[(16 * m + g + 8 * i) * kRedLd + warp] = s;
        }
    __syncthreads();
    // Warp w finishes rows 8w..8w+7, four lanes per row (two warps' partials each, summed as
    // ((w0 + w1) + (w2 + w3)) + ((w4 + w5) + (w6 + w7))): 1/RMS into the row's padding slot kWarps, read by every
    // thread after a second barrier.
    {
        const int row = 8 * warp + (lane >> 2), k = lane & 3;
        const float2 v = *reinterpret_cast<const float2*>(red + row * kRedLd + 2 * k);
        float s = v.x + v.y;
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        if (k == 0) red[row * kRedLd + kWarps] = rsqrtf(s * (1.f / kD) + kEps);
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int i = 0; i < 2; ++i) rstd[m][i] = red[(16 * m + g + 8 * i) * kRedLd + kWarps];
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

// int8 networks: norm_to_smem with int8 output: y = RMSNorm(x) · weight as int8 rows of dst (byte stride 2·kLdH),
// q = rn(y · 127 / max |y|) with scale[row] = max |y| / 127, from the FP32 values. max |y| = rstd · max |x · weight|,
// so the row maxima ride along with the sums of squares (one barrier, as norm_to_smem) and rstd cancels out of q.
// pooled (the final norm): also the token mean of y, per column. A barrier at the end, as norm_to_smem.
__device__ __forceinline__ void norm_to_smem_i8(const float (&x)[4][2][4], const float* __restrict__ weight,
                                                int8_t* dst, float* scale, float* red, int warp, int lane,
                                                float* pooled = nullptr)
{
    const int g = lane >> 2, t = lane & 3;
    float2 wt[2];
#pragma unroll
    for (int n = 0; n < 2; ++n) wt[n] = __ldg(reinterpret_cast<const float2*>(weight + 16 * warp + 8 * n + 2 * t));
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            float s = 0.f, mx = 0.f;
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                const float a = x[m][n][2 * i], c = x[m][n][2 * i + 1];
                s += a * a + c * c;
                mx = fmaxf(mx, fmaxf(fabsf(a * wt[n].x), fabsf(c * wt[n].y)));
            }
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            if (t == 0)
            {
                reinterpret_cast<float2*>(red)[(16 * m + g + 8 * i) * kRed2Ld + warp] = make_float2(s, mx);
            }
        }
    __syncthreads();
    // Warp w finishes rows 8w..8w+7, four lanes per row (two warps' partials each): (rstd, 127 / max |x · w|) into
    // the row's padding slot kWarps, read by every thread after a second barrier.
    {
        const int row = 8 * warp + (lane >> 2), k = lane & 3;
        const float4 v = *reinterpret_cast<const float4*>(red + 2 * (row * kRed2Ld + 2 * k));
        float sq = v.x + v.z, mx = fmaxf(v.y, v.w);
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        if (k == 0)
        {
            const float rstd = rsqrtf(sq * (1.f / kD) + kEps);
            reinterpret_cast<float2*>(red)[row * kRed2Ld + kWarps] =
                make_float2(rstd, mx > 0.f ? __fdividef(127.f, mx) : 0.f);
            scale[row] = rstd * mx * (1.f / 127.f);
        }
    }
    __syncthreads();
    float pc[2][2] = {};  // pooled: this thread's column sums of y
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const int row = 16 * m + g + 8 * i;
            const float2 ri = reinterpret_cast<const float2*>(red)[row * kRed2Ld + kWarps];
            const float rstd = ri.x, inv = ri.y;
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                const float a = x[m][n][2 * i] * wt[n].x, c = x[m][n][2 * i + 1] * wt[n].y;
                const uint32_t q0 = round_to_byte(a * inv), q1 = round_to_byte(c * inv);
                *reinterpret_cast<uint16_t*>(dst + row * 2 * kLdH + 16 * warp + 8 * n + 2 * t) =
                    static_cast<uint16_t>(q0 | q1 << 8);
                if (pooled)
                {
                    pc[n][0] += a * rstd;
                    pc[n][1] += c * rstd;
                }
            }
        }
    if (pooled)
    {
#pragma unroll
        for (int n = 0; n < 2; ++n)
        {
#pragma unroll
            for (int off = 4; off < 32; off <<= 1)
            {
                pc[n][0] += __shfl_xor_sync(0xffffffffu, pc[n][0], off);
                pc[n][1] += __shfl_xor_sync(0xffffffffu, pc[n][1], off);
            }
            if (g == 0)
            {
                const int col = 16 * warp + 8 * n + 2 * t;
                pooled[col] = pc[n][0] * (1.f / kTok);
                pooled[col + 1] = pc[n][1] * (1.f / kTok);
            }
        }
    }
    __syncthreads();
}

// int8 networks: the 64 FP16 rows of `buf` (128 columns, stride kLdH) become int8 in place, in the first 128 bytes
// of each row's storage (byte stride 2·kLdH), with scale[row] = max |a| / 127 and q = rn(a · 127 / max |a|)
// (src/export/int8.py's _quantize_tokens). Four threads per row: each reads 32 halves and writes 32 bytes; the
// shuffles that combine the maxima order every read of the row before its writes. Between two barriers: the FP16
// rows complete before, the int8 rows and scales before the next matmul after.
__device__ __forceinline__ void quantize_rows(__half* buf, float* scale, int tid)
{
    const int row = tid >> 2, q = tid & 3;
    const uint4* src = reinterpret_cast<const uint4*>(buf + row * kLdH + 32 * q);
    uint4 v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = src[k];
    const auto* h = reinterpret_cast<const __half2*>(v);
    __half2 m2 = __habs2(h[0]);
#pragma unroll
    for (int j = 1; j < 16; ++j) m2 = __hmax2(m2, __habs2(h[j]));
    float amax = fmaxf(__low2float(m2), __high2float(m2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float inv = amax > 0.f ? __fdividef(127.f, amax) : 0.f;
    uint32_t o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        const float2 f0 = __half22float2(h[2 * j]), f1 = __half22float2(h[2 * j + 1]);
        o[j] = round_to_byte(f0.x * inv) | round_to_byte(f0.y * inv) << 8 | round_to_byte(f1.x * inv) << 16
             | round_to_byte(f1.y * inv) << 24;
    }
    uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<int8_t*>(buf + row * kLdH) + 32 * q);
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
    if (q == 0) scale[row] = amax * (1.f / 127.f);
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

// Attention for head `warp`. qa: Q as the A operand of each row tile (already scaled by log2(e) / 4 through the
// q norm weight), kb: K as the B operand of each key tile (qk_operands); vt holds V transposed. Writes the head's
// 16 output columns of `o`.
__device__ __forceinline__ void attention(const uint32_t (&qa)[4][4], const uint32_t (&kb)[8][2], const __half* vt,
                                          const void* pair_bias, __half* o, int warp, int lane)
{
    const int g = lane >> 2, t = lane & 3;
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
        // Scores start from the pair bias (packed by to_cnet.pack_pair_bias, × log2(e)).
        float sc[8][4];
        if constexpr ((kAttnHalf & 2) != 0)
        {
            uint32_t sh[8][2];  // FP16 pairs: rows g, g + 8
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                const uint4 bv = __ldg(bias + (m * 4 + c) * 32);
                sh[2 * c][0] = bv.x;
                sh[2 * c][1] = bv.y;
                sh[2 * c + 1][0] = bv.z;
                sh[2 * c + 1][1] = bv.w;
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                mma_h(sh[j], qa[m], kb[j][0], kb[j][1]);
                const float2 lo = unh2(sh[j][0]), hi = unh2(sh[j][1]);
                sc[j][0] = lo.x;
                sc[j][1] = lo.y;
                sc[j][2] = hi.x;
                sc[j][3] = hi.y;
            }
        }
        else
        {
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
            for (int j = 0; j < 8; ++j) mma(sc[j], qa[m], kb[j][0], kb[j][1]);
        }

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
        uint32_t acch[2][2] = {};
#pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            const uint32_t pa[4] = {h2(sc[2 * s][0], sc[2 * s][1]), h2(sc[2 * s][2], sc[2 * s][3]),
                                    h2(sc[2 * s + 1][0], sc[2 * s + 1][1]),
                                    h2(sc[2 * s + 1][2], sc[2 * s + 1][3])};
#pragma unroll
            for (int d = 0; d < 2; ++d)
            {
                if constexpr ((kAttnHalf & 1) != 0)
                    mma_h(acch[d], pa, vb[s][d][0], vb[s][d][1]);
                else
                    mma(acc[d], pa, vb[s][d][0], vb[s][d][1]);
            }
        }
        if constexpr ((kAttnHalf & 1) != 0)
        {
#pragma unroll
            for (int d = 0; d < 2; ++d)
            {
                const float2 lo = unh2(acch[d][0]), hi = unh2(acch[d][1]);
                acc[d][0] = lo.x;
                acc[d][1] = lo.y;
                acc[d][2] = hi.x;
                acc[d][3] = hi.y;
            }
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

// The input planes of each token: plane_word, planes_to_row and planes_to_row_i8 (net/planes.cuh), four threads
// per token.
static_assert(kThreads == 4 * kTok, "four threads per token");

// kI8: an int8 network (EncoderParams::int8): the projections run warp_gemm_i8 on rows quantize_rows makes int8 in
// place (the embedding on int8 planes), everything else as in the FP16 kernel.
template <int kChunk, int kMinBlocks, bool kTime = false, int kMock = 0, bool kI8 = false>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    encoder_forward(__grid_constant__ const EncoderParams P, const SmallPackedPosition* __restrict__ packed,
                    const float* __restrict__ dense, unsigned char* __restrict__ out)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& s = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int b = blockIdx.x;
    // Instrumented build only: thread 0 adds the cycles since the previous mark to phase k's counter.
    long long t_mark = 0;
    if constexpr (kTime) t_mark = clock64();
    const auto mark = [&](int k) {
        if constexpr (kTime)
        {
            if (tid == 0)
            {
                const long long now = clock64();
                atomicAdd(&g_phase[k], static_cast<unsigned long long>(now - t_mark));
                t_mark = now;
            }
        }
    };

    // A projection of two n-tiles (16 output columns) of the rows in s.h, INT8 or FP16. The q/k, feed-forward up
    // and policy projections run as pairs of these, their outputs consumed before the next pair: half the live
    // outputs of one four-tile call, under the 128-register cap of two blocks per SM (A is read twice).
    const auto proj2 = [&](float (&o)[4][2][4], const void* w, const float* ws, const int (&tl)[2]) {
        if constexpr (kI8)
            warp_gemm_i8<2, 2, false>(o, reinterpret_cast<const int8_t*>(s.h), 2 * kLdH, w, ws, s.scale_h, tl, lane);
        else
        {
            zero(o);
            warp_gemm<4, 2, kChunk, 4, kMock>(o, s.h, kLdH, w, tl, lane);
        }
    };

    // ---- Input planes into s.h as [token][plane], FP16 (int8 networks: int8, 100 · plane, at the row scale 1/100).
    constexpr int kLdB = 2 * kLdH;  // the byte stride of int8 rows
    auto* const h8 = reinterpret_cast<int8_t*>(s.h);
    auto* const v8 = reinterpret_cast<int8_t*>(s.v);
    if (packed)
    {
        const auto* src = reinterpret_cast<const unsigned long long*>(packed + b);
        for (int i = tid; i < static_cast<int>(sizeof(s.packed) / 8); i += kThreads) s.packed[i] = src[i];
        __syncthreads();
        const auto& pos = *reinterpret_cast<const SmallPackedPosition*>(s.packed);
        if (kI8 && P.embed_s)
            planes_to_row_i8(pos, tid >> 2, tid & 3, P.stm_colour, h8 + (tid >> 2) * kLdB);
        else
            planes_to_row(pos, tid >> 2, tid & 3, P.stm_colour, s.h + (tid >> 2) * kLdH);
    }
    else
    {
        const float* src = dense + static_cast<size_t>(b) * kTok * kPackPlanes;
        for (int i = tid; i < kTok * 128; i += kThreads)
        {
            const int tok = i >> 7, pl = i & 127;
            const float v = pl < kPackPlanes ? src[tok * kPackPlanes + pl] : 0.f;
            if (kI8 && P.embed_s)
                h8[tok * kLdB + pl] = static_cast<int8_t>(__float2int_rn(v * 100.f));
            else
                s.h[tok * kLdH + pl] = __float2half(v);
        }
    }
    if constexpr (kI8)
    {
        if (tid < kTok) s.scale_h[tid] = 0.01f;
    }
    __syncthreads();
    mark(0);

    // ---- Embedding: x0 = planes · We + be + pos. x (FP32) stays in registers; x0 (FP16), read once per layer,
    // goes to global memory (L2): the 16 registers it took were spilled around every attention anyway. One uint4
    // per lane and row tile, [block][warp][m][lane].
    uint4* const x0s = static_cast<uint4*>(P.x0_scratch) + (static_cast<size_t>(b) * kWarps + warp) * 4 * 32 + lane;
    const int cols[2] = {2 * warp, 2 * warp + 1};
    float x[4][2][4];
    zero(x);
    if (kI8 && P.embed_s)  // an int8 network's embedding may be FP16 (no scales)
        warp_gemm_i8<2, 2, false>(x, h8, kLdB, P.embed_w, P.embed_s, s.scale_h, cols, lane);
    else
        warp_gemm<4, 2, kChunk, 4, kMock>(x, s.h, kLdH, P.embed_w, cols, lane);
    {
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
#pragma unroll
        for (int m = 0; m < 4; ++m) __stcg(x0s + m * 32, make_uint4(x0[m][0][0], x0[m][0][1], x0[m][1][0], x0[m][1][1]));
    }
    mark(1);

    for (int l = 0; l < P.layers; ++l)
    {
        const LayerParams& L = P.layer[l];
#pragma unroll
        for (int m = 0; m < 4; ++m)
        {
            const uint4 w = __ldcg(x0s + m * 32);
            const uint32_t x0m[2][2] = {{w.x, w.y}, {w.z, w.w}};
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    const float2 v = unh2(x0m[n][i]);
                    x[m][n][2 * i] += L.gain * v.x;
                    x[m][n][2 * i + 1] += L.gain * v.y;
                }
        }
        if constexpr (kI8)
            norm_to_smem_i8(x, L.norm1, h8, s.scale_h, s.red, warp, lane);
        else
            norm_to_smem(x, L.norm1, s.h, s.red, warp, lane);
        mark(2);

        // ---- Attention, head = warp. V first, transposed into the warp's own slice.
        __half* vt = s.v + warp * kHd * kLdV;
        {
            float v[4][2][4];
            zero(v);
            const int vtile[2] = {32 + 2 * warp, 33 + 2 * warp};
            if constexpr (kI8)
                warp_gemm_i8<2, 2, false>(v, h8, kLdB, L.qkv, L.qkv_s, s.scale_h, vtile, lane);
            else
                warp_gemm<4, 2, kChunk, 4, kMock>(v, s.h, kLdH, L.qkv, vtile, lane);
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                        vt[(8 * n + 2 * t + (r & 1)) * kLdV + 16 * m + g + 8 * (r >> 1)] =
                            __float2half(v[m][n][r]);
        }
        mark(3);
        {
            // K, then Q, each head-normed and packed into the score multiply's FP16 operands at once.
            uint32_t qa[4][4], kb[8][2];
            {
                float k[4][2][4];
                const int ktile[2] = {16 + 2 * warp, 17 + 2 * warp};
                proj2(k, L.qkv, L.qkv_s, ktile);
                head_norm(k, 0, L.knorm, t);
#pragma unroll
                for (int j = 0; j < 8; ++j)  // key tile j (keys 8j..8j+7): row half j % 2 of row tile j / 2
                {
                    const int m = j >> 1, i = j & 1;
                    kb[j][0] = h2(k[m][0][2 * i], k[m][0][2 * i + 1]);
                    kb[j][1] = h2(k[m][1][2 * i], k[m][1][2 * i + 1]);
                }
            }
            {
                float q[4][2][4];
                const int qtile[2] = {2 * warp, 2 * warp + 1};
                proj2(q, L.qkv, L.qkv_s, qtile);
                head_norm(q, 0, L.qnorm, t);
#pragma unroll
                for (int m = 0; m < 4; ++m)
                {
                    qa[m][0] = h2(q[m][0][0], q[m][0][1]);
                    qa[m][1] = h2(q[m][0][2], q[m][0][3]);
                    qa[m][2] = h2(q[m][1][0], q[m][1][1]);
                    qa[m][3] = h2(q[m][1][2], q[m][1][3]);
                }
            }
            __syncthreads();  // every warp has read h for its q, k and v: the attention output goes there
            mark(4);
            attention(qa, kb, vt, P.pair_bias, s.h, warp, lane);
        }
        __syncthreads();
        if (kI8 && L.out_s)  // an int8 network's out may be FP16 (no scales)
        {
            quantize_rows(s.h, s.scale_h, tid);
            __syncthreads();
        }
        mark(5);
        if (kI8 && L.out_s)
            warp_gemm_i8<2, 2, true>(x, h8, kLdB, L.out, L.out_s, s.scale_h, cols, lane);
        else
            warp_gemm<4, 2, kChunk, 4, kMock>(x, s.h, kLdH, L.out, cols, lane);
        mark(6);

        // ---- SwiGLU feed-forward in two halves of 128 hidden columns (16 gate + 16 up per warp),
        // each through v (free until the next layer's V) and then its half of the down projection.
        if constexpr (kI8)
            norm_to_smem_i8(x, L.norm2, h8, s.scale_h, s.red, warp, lane);
        else
            norm_to_smem(x, L.norm2, s.h, s.red, warp, lane);
        mark(7);
#pragma unroll
        for (int half = 0; half < 2; ++half)
        {
            const int c0 = 16 * half + 2 * warp;
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                float u[4][2][4];  // the gate and up columns of the same 8 hidden units
                const int utile[2] = {c0 + n, 32 + c0 + n};
                proj2(u, L.up, L.up_s, utile);
                const int col = 16 * warp + 8 * n + 2 * t;  // within the half
#pragma unroll
                for (int m = 0; m < 4; ++m)
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                    {
                        const int row = 16 * m + g + 8 * i;
                        *reinterpret_cast<uint32_t*>(s.v + row * kLdH + col) =
                            h2(silu(u[m][0][2 * i]) * u[m][1][2 * i], silu(u[m][0][2 * i + 1]) * u[m][1][2 * i + 1]);
                    }
            }
            __syncthreads();
            if (kI8 && L.down_s)  // an int8 network's down may be FP16 (no scales)
            {
                quantize_rows(s.v, s.scale_v, tid);  // per token and hidden half (int8.py: down's two groups)
                __syncthreads();
            }
            mark(8);
            if (kI8 && L.down_s)
                warp_gemm_i8<2, 2, true, 4>(x, v8, kLdB, L.down, L.down_s, s.scale_v, cols, lane, 2 * half);
            else
                warp_gemm<4, 2, kChunk, 8, kMock>(x, s.v, kLdH, L.down, cols, lane, 4 * half);
            // Before the second half overwrites v; after it, the next norm's barrier comes first.
            if (half == 0) __syncthreads();
            mark(9);
        }
    }

    // ---- Final norm into s.h, and the token mean of the normed stream (value input).
    if constexpr (kI8)
    {
        norm_to_smem_i8(x, P.norm, h8, s.scale_h, s.red, warp, lane, s.pooled);
    }
    else
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
        __syncthreads();
    }
    mark(10);

    // ---- Policy: the q and k projections (q pre-scaled by 1/sqrt(d) in the file). k into v at once (nothing
    // reads v after the last down projection, which the final norm's barrier follows), q into h once every warp
    // has read h.
#pragma unroll
    for (int part = 1; part >= 0; --part)
    {
        float a[4][2][4];
        const int ptile[2] = {16 * part + 2 * warp, 16 * part + 2 * warp + 1};
        proj2(a, P.pol_w, P.pol_s, ptile);
        if (part == 0) __syncthreads();
#pragma unroll
        for (int n = 0; n < 2; ++n)
        {
            const int col = 8 * ptile[n] + 2 * t;
            const float2 bb = __ldg(reinterpret_cast<const float2*>(P.pol_b + col));
            __half* dst = part == 0 ? s.h + col : s.v + (col - kD);
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    const int row = 16 * m + g + 8 * i;
                    *reinterpret_cast<uint32_t*>(dst + row * kLdH) =
                        h2(a[m][n][2 * i] + bb.x, a[m][n][2 * i + 1] + bb.y);
                }
        }
    }
    __syncthreads();
    mark(11);

    // Scores for to-tokens 8·warp..8·warp+7: ft[from][to] = q_from · k_to and
    // pt[promotion][to] = promo · k_to (promotion rows 0-3 of one extra row tile).
    float ft[4][4] = {}, pt[4] = {};
    {
        const auto* promo = static_cast<const uint32_t*>(P.promo);  // [4][64] FP16 pairs
        const __half* arow = s.h + (lane & 15) * kLdH + (lane >> 4) * 8;
        const __half* krow = s.v + (8 * warp + g) * kLdH + 2 * t;
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks)
        {
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow + 16 * ks);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + 16 * ks + 8);
#pragma unroll
            for (int m = 0; m < 4; ++m)
            {
                uint32_t af[4];
                ldsm4(af, arow + 16 * m * kLdH + 16 * ks);
                mma(ft[m], af, b0, b1);
            }
            const uint32_t pa[4] = {g < 4 ? __ldg(promo + g * 64 + 8 * ks + t) : 0u, 0u,
                                    g < 4 ? __ldg(promo + g * 64 + 8 * ks + 4 + t) : 0u, 0u};
            mma(pt, pa, b0, b1);
        }
    }
    __syncthreads();
    float* sc = reinterpret_cast<float*>(s.h);  // [68][kLdS] over h and v: from-token rows, then promotions
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
    mark(12);
    auto* policy = reinterpret_cast<__half*>(out + static_cast<size_t>(b) * kRecordBytes);
    for (int i = tid; i < kPolicySize; i += kThreads)
    {
        const unsigned code = __ldg(P.vocab + i);
        const int to = (code >> 6) & 63;
        policy[i] = __float2half(sc[(code & 63) * kLdS + to] + sc[(kTok + ((code >> 12) & 3)) * kLdS + to]);
    }
    mark(13);

    // ---- Value heads: MLPs on the pooled vector (first layers stored [mlp][in][out]).
    // First layers: an item is two adjacent units over a quarter of the inputs (32 half2 loads, all in flight;
    // one unit and all 128 inputs per thread was 128 dependent rounds of 2-byte loads from L2, 5% of the kernel),
    // the four quarters in adjacent lanes summed by shuffles. n_mlp · 256 items: whole warps.
    const int n_mlp = P.n_wdl + 1 + P.mlhq_from_q;
    for (int item = tid; item < n_mlp * kD * 2; item += kThreads)
    {
        const int u = 2 * (item >> 2), quarter = item & 3;  // units u, u + 1 (same MLP: kD is even)
        const auto* w = reinterpret_cast<const __half2*>(static_cast<const __half*>(P.mlp1_w) + (u >> 7) * kD * kD
                                                         + quarter * (kD / 4) * kD + (u & 127));
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int k = 0; k < kD / 4; ++k)
        {
            const float2 wv = __half22float2(__ldg(w + k * (kD / 2)));
            const float pk = s.pooled[quarter * (kD / 4) + k];
            a0 += wv.x * pk;
            a1 += wv.y * pk;
        }
        a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        a0 += __shfl_xor_sync(0xffffffffu, a0, 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
        if (quarter == 0)
        {
            s.hidden[u] = silu(a0 + __ldg(P.mlp1_b + u));
            s.hidden[u + 1] = silu(a1 + __ldg(P.mlp1_b + u + 1));
        }
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
            const float scale = P.wdl_weight[k] / (e0 + e1 + e2);
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
    mark(14);
}

template <int kChunk, int kMinBlocks, bool kTime = false, int kMock = 0, bool kI8 = false>
cudaError_t launch(const EncoderParams& params, const SmallPackedPosition* packed, const float* dense,
                   int batch, void* out, cudaStream_t stream)
{
    static const cudaError_t setup = [] {
        const cudaError_t e = cudaFuncSetAttribute(encoder_forward<kChunk, kMinBlocks, kTime, kMock, kI8>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(sizeof(Smem)));
        if (e != cudaSuccess) return e;
        return cudaFuncSetAttribute(encoder_forward<kChunk, kMinBlocks, kTime, kMock, kI8>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    }();
    if (setup != cudaSuccess) return setup;
    encoder_forward<kChunk, kMinBlocks, kTime, kMock, kI8><<<batch, kThreads, sizeof(Smem), stream>>>(
        params, packed, dense, static_cast<unsigned char*>(out));
    return cudaGetLastError();
}

bool kernel_timing()
{
    static const bool on = [] {
        const char* env = std::getenv("CHESSRL_KERNEL_TIMING");
        return env != nullptr && std::atoi(env) == 1;
    }();
    return on;
}

// CHESSRL_KERNEL_MOCK=1..7: the projection loop's timing experiments (warp_gemm's kMock; wrong outputs).
int kernel_mock()
{
    static const int mock = [] {
        const char* env = std::getenv("CHESSRL_KERNEL_MOCK");
        const int v = env ? std::atoi(env) : 0;
        return v >= 0 && v <= 7 ? v : 0;
    }();
    return mock;
}

// With the default FP16 accumulation the kernel keeps two blocks on each SM (at most 128 registers
// per thread, the rest spilled to the stack): a block's barriers and memory waits are covered by the
// other block, +33% on its own on an RTX 4090 (1.41M to 1.88M positions/s).
// CHESSRL_KERNEL_BLOCKS=1 selects the one-block build.
int kernel_blocks()
{
    static const int blocks = [] {
        const char* env = std::getenv("CHESSRL_KERNEL_BLOCKS");
        return env != nullptr && std::atoi(env) == 1 ? 1 : 2;
    }();
    return blocks;
}

// The projection matrix multiplies sum in FP16 (whole sums by default): CHESSRL_FP16_ACC=0 keeps
// FP32 accumulators, N > 0 sums N terms (32, 64, or 256 for whole sums) in FP16 between FP32
// additions. The score multiplies (attention, policy logits) and the residual stream stay FP32.
// Measured: parity with PyTorch unchanged to within KL 3e-5, kernel +29%, and in a
// real run the trainer gets the GPU time it frees (3.48 -> 4.86 steps/s next to self-play).
int fp16_chunk()
{
    static const int chunk = [] {
        const char* env = std::getenv("CHESSRL_FP16_ACC");
        const int terms = env ? std::atoi(env) : 256;
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
    // An int8 network: INT8 projections, two blocks per SM (CHESSRL_KERNEL_BLOCKS=1: one, all 255 registers; the
    // accumulation switches do not apply).
    if (params.int8)
    {
        if (kernel_timing()) return launch<8, 2, true, 0, true>(params, packed, dense, batch, out, stream);
        return kernel_blocks() == 2 ? launch<8, 2, false, 0, true>(params, packed, dense, batch, out, stream)
                                    : launch<8, 1, false, 0, true>(params, packed, dense, batch, out, stream);
    }
    if (kernel_timing()) return launch<8, 2, true>(params, packed, dense, batch, out, stream);
    switch (kernel_mock())
    {
    case 1: return launch<8, 2, false, 1>(params, packed, dense, batch, out, stream);
    case 2: return launch<8, 2, false, 2>(params, packed, dense, batch, out, stream);
    case 3: return launch<8, 2, false, 3>(params, packed, dense, batch, out, stream);
    case 4: return launch<8, 2, false, 4>(params, packed, dense, batch, out, stream);
    case 6: return launch<8, 2, false, 6>(params, packed, dense, batch, out, stream);
    case 7: return launch<8, 2, false, 7>(params, packed, dense, batch, out, stream);
    default: break;
    }
    // Every accumulation mode honours CHESSRL_KERNEL_BLOCKS (two blocks per SM by default), so the modes compare
    // at the same occupancy: the 32-bit ones then show what 32-bit sums cost under the 128-register cap.
    const bool two = kernel_blocks() == 2;
    switch (fp16_chunk())
    {
    case 1: return two ? launch<1, 2>(params, packed, dense, batch, out, stream)
                       : launch<1, 1>(params, packed, dense, batch, out, stream);
    case 2: return two ? launch<2, 2>(params, packed, dense, batch, out, stream)
                       : launch<2, 1>(params, packed, dense, batch, out, stream);
    case 8: return two ? launch<8, 2>(params, packed, dense, batch, out, stream)
                       : launch<8, 1>(params, packed, dense, batch, out, stream);
    default: return two ? launch<0, 2>(params, packed, dense, batch, out, stream)
                        : launch<0, 1>(params, packed, dense, batch, out, stream);
    }
}

std::vector<unsigned long long> kernel_phase_cycles(bool reset)
{
    std::vector<unsigned long long> out(kPhases, 0ull);
    cudaMemcpyFromSymbol(out.data(), g_phase, sizeof(unsigned long long) * kPhases);
    if (reset)
    {
        const std::vector<unsigned long long> zeros(kPhases, 0ull);
        cudaMemcpyToSymbol(g_phase, zeros.data(), sizeof(unsigned long long) * kPhases);
    }
    return out;
}

std::vector<std::string> kernel_phase_names()
{
    return std::vector<std::string>(kPhaseNames, kPhaseNames + kPhases);
}

} // namespace chessrl::net
