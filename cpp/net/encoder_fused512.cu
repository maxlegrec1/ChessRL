// Fused forward pass of INT8 encoder networks of d_model 512, 8 heads of 64 and d_ff 768 (src/models/encoder.py; any
// depth up to kMaxLayers): one 16-warp thread block per position runs the whole network, from the packed position to
// the policy logits and value outputs, in a single launch. Same output records as encoder_kernel.cu.
//
// Arithmetic (the INT8 specification src/export/int8.py, at these sizes): every projection is m16n8k32 INT8 with
// INT32 sums, weights with one scale per output row, activations quantized per token where they are produced: the
// norms' FP32 outputs, the attention output per token and head (64 columns), the SwiGLU output per token and
// 128-column chunk. The embedding's planes enter exactly (100 · plane, scale 1/100). The residual stream is FP16, as
// training's (FP16_RESIDUAL); norms, scores and softmax are FP32; P · V sums in FP16 (P <= 1).
//
// Layout: warp w owns columns 32w..32w+31 of the residual stream for all 64 tokens and keeps them in registers in the
// mma.sync accumulator layout (x[m][n][i]: token 16m + g + 8i, column 32w + 8n + 2t and + 1; lane = 4g + t). A
// operands are INT8 rows in shared memory whose 16-byte chunks are swizzled by the row (swz), so the 8 rows of an
// ldmatrix hit distinct banks without padding; weights stream from L2 as packed B fragments, one k64 step ahead.
// The out and down projections are deferred: the attention output (all heads) and the SwiGLU output (all chunks) stay
// in shared memory as INT8, and each sub-layer adds into the residual once, in FP32, rounded once to FP16.
//
// Attention sub-layer: warps 0-11 compute each head's q, k, v (24 n-tiles, 2 each, all 64 rows) into a one-head
// buffer while warps 12-15 run the previous head's attention, one 16-query tile each; named barriers hand the buffer
// back and forth (bar.arrive / bar.sync). The QK-norm is folded into the scores: the q, k projections' epilogue
// writes q and k' = k · qnorm · knorm with each row's sum of squares, and s = (q · k') / (RMS(q) RMS(k)) + bias.
// Feed-forward: six chunks of 128 SwiGLU columns (gate and up rows interleaved per 8 in the file); a chunk's
// per-token maxima meet in shared memory through mbarriers (four buffers), so a warp quantizes chunk c - 1 after its
// own chunk c instead of waiting at a block barrier.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "net/encoder_fused512.hpp"
#include "net/planes.cuh"

namespace chessrl::net
{
namespace
{

using mcts::single::SmallPackedPosition;
using mcts::single::kPackPlanes;

constexpr int kTok = 64, kD = kF512D, kHeads = kF512Heads, kHd = kD / kHeads, kFf = kF512Ff;
constexpr int kChunk = 128, kChunks = kFf / kChunk;
constexpr int kWarps = 16, kThreads = 32 * kWarps, kQkvWarps = 12;
constexpr int kPolChunk = 128;  // policy q / k columns per pass
constexpr float kEps = 1e-6f;
static_assert(kHd == 64 && kFf % kChunk == 0 && kWarps * 32 == kD, "the layout's sizes");

// Shared memory (bytes). A: the planes, then each norm's INT8 output. Region B: in the attention sub-layer o (INT8,
// all heads) and one head's FP16 q, k', v; in the feed-forward g (INT8, all chunks) and the chunks' FP16 row-max
// partials; the norms' partials while it holds nothing else; after the layers the pooled vector, the policy's q and
// k columns of one pass, then the scores and the value heads' hidden units.
constexpr int kOffA = 0;
constexpr int kOffB = kTok * kD;
constexpr int kOffO = kOffB;
constexpr int kOffQkv = kOffO + kTok * kD;
constexpr int kQkvBytes = 3 * kTok * kHd * 2;
constexpr int kOffG = kOffB;
constexpr int kGBufs = 4;
constexpr int kOffRedG = kOffG + kTok * kFf;
constexpr int kOffRed = kOffB;
constexpr int kOffPooled = kOffB + 8192;
constexpr int kOffPolQ = kOffB + 16384;
constexpr int kOffPolK = kOffPolQ + kTok * kPolChunk * 2;
constexpr int kOffScores = kOffPolQ;
constexpr int kLdS = kTok + 4;  // FP32 score rows
constexpr int kOffHidden = kOffB + 36864;
constexpr int kSizeB = kTok * kD + kQkvBytes;
static_assert(kTok * kFf + kGBufs * kTok * kWarps * 2 <= kSizeB, "the feed-forward fits region B");
static_assert(kOffPolK + kTok * kPolChunk * 2 <= kOffB + kSizeB, "the policy pass fits region B");
static_assert(kOffScores + (kTok + 4) * kLdS * 4 <= kOffHidden && kOffHidden + 3 * kD * 4 <= kOffB + kSizeB,
              "scores and hidden units fit region B");
static_assert(kOffB + sizeof(SmallPackedPosition) <= kOffPooled, "the packed position fits");
constexpr int kOffScale = kOffB + kSizeB;

struct Scales
{
    float a[kTok];           // the last norm output's token scales
    float o[kTok][kHeads];   // o's token-and-head scales
    float g[kTok][kChunks];  // g's token-and-chunk scales
    float2 rows[kTok];       // the norms' (rstd, 127 / max |x w|)
    float kfold[kHd];        // the layer's qnorm · knorm
    float heads[16];         // the value MLPs' outputs
};
constexpr int kOffMbar = kOffScale + static_cast<int>(sizeof(Scales));
// One head's q and k row sums of squares, [q / k][64 rows][4 warps' 16 columns]; the attention warps turn each row's
// into its 1/RMS in place (element 0).
constexpr int kOffQkPart = kOffMbar + kGBufs * 8;
constexpr int kSmem = kOffQkPart + 2 * kTok * 4 * 4;
static_assert(kOffMbar % 8 == 0, "mbarrier alignment");
static_assert(kSmem <= 101376, "one block per SM");

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

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p)
{
    const auto a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p)
{
    const auto a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

// c += a (16x16, row major) · b (16x8, column major); FP16 inputs, FP32 accumulators.
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The same with FP16 accumulators (twice the rate on GeForce cards): c[0] row g, c[1] row g + 8, two halves each.
__device__ __forceinline__ void mma_h(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, "
        "{%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// c += a (16x32, row major) · b (32x8, column major); INT8 inputs, INT32 sums. The A fragment has the byte layout of
// m16n8k16's FP16 one, so ldmatrix loads it from 16 rows of 32 bytes.
__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Named barriers over the whole block: producers arrive (no wait), consumers sync.
__device__ __forceinline__ void bar_sync(int id)
{
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "n"(kThreads) : "memory");
}

__device__ __forceinline__ void bar_arrive(int id)
{
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "n"(kThreads) : "memory");
}
constexpr int kBarAttn = 1;  // the four attention warps among themselves
constexpr int kBarQkvFull = 2, kBarQkvFree = 3;

// mbarriers: split arrive and wait, so a warp can arrive for one feed-forward chunk and wait for an earlier one.
__device__ __forceinline__ uint32_t smem_addr(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count)
{
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(smem_addr(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity)
{
    uint32_t done;
    do
    {
        asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done)
                     : "r"(smem_addr(bar)), "r"(parity)
                     : "memory");
    } while (!done);
}

// Round to the nearest integer (ties to even) and keep the low byte, without the conversion unit: exact for
// |v| < 2^22.
__device__ __forceinline__ uint32_t round_to_byte(float v)
{
    return __float_as_uint(v + 12582912.f) & 0xFFu;
}

__device__ __forceinline__ float exp2_approx(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float tanh_approx(float x)
{
    float y;
    asm("tanh.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// x · sigmoid(x) = 0.5 x (1 + tanh(x / 2)): one MUFU operation.
__device__ __forceinline__ float silu_fast(float x)
{
    const float h = 0.5f * x;
    return fmaf(h, tanh_approx(h), h);
}

// Byte offset of (row, byte) in a buffer of `stride`-byte rows (a multiple of 128) whose 16-byte chunks are swizzled
// by the row: chunk c of row r lives at c ^ (r & 7) within its 128-byte group.
__device__ __forceinline__ int swz(int row, int byte, int stride)
{
    return row * stride + (byte & ~127) + ((((byte >> 4) ^ row) & 7) << 4) + (byte & 15);
}

// The swizzled offset, within a 128-byte group of an A row, of the ldmatrix chunk for k32 sub-step s of a k64 step of
// the given parity (an ldmatrix row's low 3 bits are lane & 7 for every m-tile: a per-lane constant).
__device__ __forceinline__ int sw_off(int parity, int s, int lane)
{
    return (((4 * parity + 2 * s + (lane >> 4)) ^ lane) & 7) << 4;
}

// The first k64 step's B fragments of n-tiles tile[] (ks k64 steps each), for gemm64: issued early (before a barrier
// or an epilogue) to hide the L2 latency.
template <int NT>
__device__ __forceinline__ void load_b(uint4 (&b)[NT], const uint4* __restrict__ w, int ks, const int (&tile)[NT],
                                       int lane)
{
#pragma unroll
    for (int n = 0; n < NT; ++n) b[n] = __ldg(w + static_cast<size_t>(tile[n] * ks) * 32 + lane);
}

// One k64 step (kstep, of static parity kPar) of c[m][n] += A · Bᵀ over the 4 m-tiles.
template <int NT, int kPar>
__device__ __forceinline__ void mma_step(int (&c)[4][NT][4], const unsigned char* arow, int stride, int kstep,
                                         const uint4 (&b)[NT], int lane)
{
    const unsigned char* base = arow + 128 * (kstep >> 1);
#pragma unroll
    for (int s = 0; s < 2; ++s)
    {
        const int off = sw_off(kPar, s, lane);
#pragma unroll
        for (int m = 0; m < 4; ++m)
        {
            uint32_t af[4];
            ldsm4(af, base + 16 * m * stride + off);
#pragma unroll
            for (int n = 0; n < NT; ++n) mma_s8(c[m][n], af, s ? b[n].z : b[n].x, s ? b[n].w : b[n].y);
        }
    }
}

// c[m][n] = A (64 rows, INT8, swizzled rows of `stride` bytes) · W[tile[n]]ᵀ over the ks (even) k64 steps of each
// n-tile. b holds step 0 on entry (load_b), junk on exit; later steps load one step ahead into alternating buffers.
template <int NT>
__device__ __forceinline__ void gemm64(int (&c)[4][NT][4], const unsigned char* a, int stride,
                                       const uint4* __restrict__ w, int ks, const int (&tile)[NT], int lane,
                                       uint4 (&b)[NT])
{
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int r = 0; r < 4; ++r) c[m][n][r] = 0;
    const uint4* wp[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) wp[n] = w + static_cast<size_t>(tile[n] * ks) * 32 + lane;
    const unsigned char* arow = a + (lane & 15) * stride;
    uint4 bn[NT];
#pragma unroll 1
    for (int p = 0; p < ks; p += 2)
    {
#pragma unroll
        for (int n = 0; n < NT; ++n) bn[n] = __ldg(wp[n] + 32);
        mma_step<NT, 0>(c, arow, stride, p, b, lane);
        const int adv = p + 2 < ks ? 64 : 32;  // the last pair re-reads step p + 1 rather than run past the tile
#pragma unroll
        for (int n = 0; n < NT; ++n)
        {
            wp[n] += adv;
            b[n] = __ldg(wp[n]);
        }
        mma_step<NT, 1>(c, arow, stride, p + 1, bn, lane);
    }
}

// x0's per-block scratch: this warp's 32 words per lane as 8 uint4 ([block][warp][8][lane]), coalesced.
__device__ __forceinline__ uint4* x0_slot(void* scratch, int warp, int lane)
{
    return static_cast<uint4*>(scratch) + (static_cast<size_t>(blockIdx.x) * kWarps + warp) * 8 * 32 + lane;
}

// x (+ gain · x0, rounded to FP16) -> RMSNorm · weight as INT8 rows of A with token scales: q = rn(x · w · 127 /
// max |x · w|), scale = rstd · max |x · w| / 127 (rstd cancels out of q). kPool (the final norm): also the token mean
// of the FP32 output y = x · rstd · w per column, into the pooled vector. Ends with a barrier.
template <bool kX0, bool kPool>
__device__ __forceinline__ void norm_i8(uint32_t (&x)[4][4][2], const uint4* x0, float gain,
                                        const float* __restrict__ weight, unsigned char* smem, int warp, int lane,
                                        int tid)
{
    const int g = lane >> 2, t = lane & 3;
    auto* red = reinterpret_cast<float2*>(smem + kOffRed);
    auto* sc = reinterpret_cast<Scales*>(smem + kOffScale);
    float2 wt[4];
#pragma unroll
    for (int n = 0; n < 4; ++n) wt[n] = __ldg(reinterpret_cast<const float2*>(weight + 32 * warp + 8 * n + 2 * t));
    if constexpr (kX0)
    {
        uint4 z[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) z[k] = x0[32 * k];
        const auto* zw = reinterpret_cast<const uint32_t*>(z);
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int n = 0; n < 4; ++n)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    const float2 v = unh2(x[m][n][i]), e = unh2(zw[(m * 4 + n) * 2 + i]);
                    x[m][n][i] = h2(__fadd_rn(v.x, __fmul_rn(gain, e.x)), __fadd_rn(v.y, __fmul_rn(gain, e.y)));
                }
    }
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const int row = 16 * m + g + 8 * i;
            float s = 0.f, mx = 0.f;
#pragma unroll
            for (int n = 0; n < 4; ++n)
            {
                const float2 v = unh2(x[m][n][i]);
                s += v.x * v.x + v.y * v.y;
                mx = fmaxf(mx, fmaxf(fabsf(__fmul_rn(v.x, wt[n].x)), fabsf(__fmul_rn(v.y, wt[n].y))));
            }
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            if (t == 0) red[row * kWarps + warp] = make_float2(s, mx);
        }
    __syncthreads();
    {
        const int row = tid >> 3, k = tid & 7;
        const float4 v = *reinterpret_cast<const float4*>(red + row * kWarps + 2 * k);
        float sq = v.x + v.z, mx = fmaxf(v.y, v.w);
#pragma unroll
        for (int off = 1; off < 8; off <<= 1)
        {
            sq += __shfl_xor_sync(0xffffffffu, sq, off);
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        }
        if (k == 0)
        {
            const float rstd = rsqrtf(sq * (1.f / kD) + kEps);
            sc->rows[row] = make_float2(rstd, mx > 0.f ? 127.f / mx : 0.f);
            sc->a[row] = __fmul_rn(__fmul_rn(rstd, mx), 1.f / 127.f);
        }
    }
    __syncthreads();
    unsigned char* a = smem + kOffA;
    float pc[4][2] = {};  // kPool: this thread's column sums of y
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const int row = 16 * m + g + 8 * i;
            const float2 ri = sc->rows[row];
#pragma unroll
            for (int n = 0; n < 4; ++n)
            {
                const float2 v = unh2(x[m][n][i]);
                const float y0 = __fmul_rn(v.x, wt[n].x), y1 = __fmul_rn(v.y, wt[n].y);
                *reinterpret_cast<uint16_t*>(a + swz(row, 32 * warp + 8 * n + 2 * t, kD)) = static_cast<uint16_t>(
                    round_to_byte(__fmul_rn(y0, ri.y)) | round_to_byte(__fmul_rn(y1, ri.y)) << 8);
                if constexpr (kPool)
                {
                    pc[n][0] += y0 * ri.x;
                    pc[n][1] += y1 * ri.x;
                }
            }
        }
    if constexpr (kPool)
    {
        auto* pooled = reinterpret_cast<float*>(smem + kOffPooled);
#pragma unroll
        for (int n = 0; n < 4; ++n)
        {
#pragma unroll
            for (int off = 4; off < 32; off <<= 1)
            {
                pc[n][0] += __shfl_xor_sync(0xffffffffu, pc[n][0], off);
                pc[n][1] += __shfl_xor_sync(0xffffffffu, pc[n][1], off);
            }
            if (g == 0)
            {
                pooled[32 * warp + 8 * n + 2 * t] = pc[n][0] * (1.f / kTok);
                pooled[32 * warp + 8 * n + 2 * t + 1] = pc[n][1] * (1.f / kTok);
            }
        }
    }
    __syncthreads();
}

// Attention warps (query tile qt): the q / k rows' 1/RMS (QK-norm as row scales), scores s = (q · k') rq rk + bias,
// softmax (exp2: log2(e) is folded into kfold and the bias), P V in FP16 sums; o rounded to FP16 and quantized per
// token over the head's 64 columns into o's columns 64h.., the scale into Scales::o[row][h].
__device__ __forceinline__ void attention_head(int h, int qt, const uint4* __restrict__ pair_bias,
                                               unsigned char* smem, int lane)
{
    const int g = lane >> 2, t = lane & 3;
    unsigned char* qb = smem + kOffQkv;
    unsigned char* kb = qb + kTok * kHd * 2;
    unsigned char* vb = kb + kTok * kHd * 2;
    auto* sc = reinterpret_cast<Scales*>(smem + kOffScale);
    uint4 bv[4];  // the pair bias, consumed after the scores
    {
        const uint4* bias = pair_bias + ((h * 4 + qt) * 4) * 32 + lane;
#pragma unroll
        for (int c = 0; c < 4; ++c) bv[c] = __ldg(bias + 32 * c);
    }
    // Lanes 0-15 turn q row 16qt + lane's sum of squares into its 1/RMS, lanes 16-31 k's.
    auto* part = reinterpret_cast<float*>(smem + kOffQkPart);
    {
        float* pr = part + ((lane >> 4) * kTok + 16 * qt + (lane & 15)) * 4;
        const float4 p = *reinterpret_cast<const float4*>(pr);
        pr[0] = rsqrtf(((p.x + p.y) + (p.z + p.w)) * (1.f / kHd) + kEps);
    }
    asm volatile("bar.sync %0, 128;\n" ::"n"(kBarAttn) : "memory");
    float s[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) s[n][r] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
    {
        uint32_t af[4];
        ldsm4(af, qb + swz(16 * qt + (lane & 15), 32 * kk + 16 * (lane >> 4), 128));
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint32_t bf[4];
            ldsm4(bf, kb + swz(16 * j + (lane & 7) + 8 * (lane >> 4), 32 * kk + 16 * ((lane >> 3) & 1), 128));
            mma(s[2 * j], af, bf[0], bf[1]);
            mma(s[2 * j + 1], af, bf[2], bf[3]);
        }
    }
    {
        const float rq0 = part[(16 * qt + g) * 4], rq1 = part[(16 * qt + g + 8) * 4];
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            const auto* bw = reinterpret_cast<const uint32_t*>(&bv[c]);
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const float2 f = unh2(bw[j]);  // elements 2j, 2j + 1: key tile 2c + j / 2, registers 2(j % 2), +1
                const int n = 2 * c + j / 2, r = 2 * (j % 2);
                const float rq = r ? rq1 : rq0;
                const float rk0 = part[(kTok + 8 * n + 2 * t) * 4], rk1 = part[(kTok + 8 * n + 2 * t + 1) * 4];
                s[n][r] = __fmaf_rn(__fmul_rn(s[n][r], rk0), rq, f.x);
                s[n][r + 1] = __fmaf_rn(__fmul_rn(s[n][r + 1], rk1), rq, f.y);
            }
        }
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < 8; ++n)
    {
        mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
        mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
    {
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    float l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int n = 0; n < 8; ++n)
    {
        s[n][0] = exp2_approx(s[n][0] - mx0);
        s[n][1] = exp2_approx(s[n][1] - mx0);
        s[n][2] = exp2_approx(s[n][2] - mx1);
        s[n][3] = exp2_approx(s[n][3] - mx1);
        l0 += s[n][0] + s[n][1];
        l1 += s[n][2] + s[n][3];
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
    {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    uint32_t pa[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
    {
        pa[kk][0] = h2(s[2 * kk][0], s[2 * kk][1]);
        pa[kk][1] = h2(s[2 * kk][2], s[2 * kk][3]);
        pa[kk][2] = h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
        pa[kk][3] = h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
    }
    uint32_t oh[8][2];  // P V, FP16 sums: oh[n][0] row g, oh[n][1] row g + 8
#pragma unroll
    for (int n = 0; n < 8; ++n) oh[n][0] = oh[n][1] = 0u;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint32_t bf[4];
            ldsm4t(bf, vb + swz(16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1), 32 * j + 16 * (lane >> 4), 128));
            mma_h(oh[2 * j], pa[kk], bf[0], bf[1]);
            mma_h(oh[2 * j + 1], pa[kk], bf[2], bf[3]);
        }
    const float il0 = 1.f / l0, il1 = 1.f / l1;
    float o[8][4];
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int n = 0; n < 8; ++n)
    {
        const float2 f0 = unh2(oh[n][0]), f1 = unh2(oh[n][1]);
        const float2 r0 = unh2(h2(f0.x * il0, f0.y * il0)), r1 = unh2(h2(f1.x * il1, f1.y * il1));
        o[n][0] = r0.x;
        o[n][1] = r0.y;
        o[n][2] = r1.x;
        o[n][3] = r1.y;
        a0 = fmaxf(a0, fmaxf(fabsf(o[n][0]), fabsf(o[n][1])));
        a1 = fmaxf(a1, fmaxf(fabsf(o[n][2]), fabsf(o[n][3])));
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
    {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, off));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, off));
    }
    const float inv0 = a0 > 0.f ? 127.f / a0 : 0.f, inv1 = a1 > 0.f ? 127.f / a1 : 0.f;
    const int r0 = 16 * qt + g, r1 = r0 + 8;
    unsigned char* ob = smem + kOffO;
#pragma unroll
    for (int n = 0; n < 8; ++n)
    {
        const int col = 64 * h + 8 * n + 2 * t;
        *reinterpret_cast<uint16_t*>(ob + swz(r0, col, kD)) = static_cast<uint16_t>(
            round_to_byte(__fmul_rn(o[n][0], inv0)) | round_to_byte(__fmul_rn(o[n][1], inv0)) << 8);
        *reinterpret_cast<uint16_t*>(ob + swz(r1, col, kD)) = static_cast<uint16_t>(
            round_to_byte(__fmul_rn(o[n][2], inv1)) | round_to_byte(__fmul_rn(o[n][3], inv1)) << 8);
    }
    if (t == 0)
    {
        sc->o[r0][h] = __fmul_rn(a0, 1.f / 127.f);
        sc->o[r1][h] = __fmul_rn(a1, 1.f / 127.f);
    }
}

// One group (kGp k64 steps from kstep0, kstep0's parity kPar) of a projection pass: per m-tile INT32 sums, then
// acc += sums · the rows' group scales.
template <int kK, int kGp, int kPar>
__device__ __forceinline__ void proj_group(float (&acc)[4][2][4], const unsigned char* arow, const float* sa,
                                           int groups, int gr, int kstep0, const uint4 (&b)[2][kGp], int lane)
{
    const int g = lane >> 2;
#pragma unroll
    for (int m = 0; m < 4; ++m)
    {
        int c[2][4];
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int r = 0; r < 4; ++r) c[n][r] = 0;
#pragma unroll
        for (int p = 0; p < kGp; ++p)
        {
            const unsigned char* base = arow + 16 * m * kK + 128 * ((kstep0 + p) >> 1);
#pragma unroll
            for (int s = 0; s < 2; ++s)
            {
                uint32_t af[4];
                ldsm4(af, base + sw_off((kPar + p) & 1, s, lane));
#pragma unroll
                for (int n = 0; n < 2; ++n) mma_s8(c[n], af, s ? b[n][p].z : b[n][p].x, s ? b[n][p].w : b[n][p].y);
            }
        }
        const float s0 = sa[(16 * m + g) * groups + gr], s1 = sa[(16 * m + g + 8) * groups + gr];
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[m][n][r] = __fmaf_rn(__int2float_rn(c[n][r]), r >> 1 ? s1 : s0, acc[m][n][r]);
    }
}

// x += (Σ_groups A_group · W_groupᵀ · sa[row][group]) · s_w, for this warp's 32 columns in two passes of 16. A: INT8
// [64][kK] swizzled; a group is kGp k64 steps (an even number of groups); B one group ahead.
template <int kK, int kGp>
__device__ __forceinline__ void project_into(uint32_t (&x)[4][4][2], const unsigned char* a, const float* sa,
                                             const uint4* __restrict__ w, const float* __restrict__ sw, int warp,
                                             int lane)
{
    constexpr int kKs = kK / 64, kGroups = kKs / kGp;
    static_assert(kGroups % 2 == 0, "groups in pairs");
    const int t = lane & 3;
    const unsigned char* arow = a + (lane & 15) * kK;
#pragma unroll  // static indices into x: a dynamic one puts the residual in local memory
    for (int np = 0; np < 2; ++np)
    {
        const int tile0 = 4 * warp + 2 * np;
        float acc[4][2][4];
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[m][n][r] = 0.f;
        const uint4* wp[2] = {w + static_cast<size_t>(tile0 * kKs) * 32 + lane,
                              w + static_cast<size_t>((tile0 + 1) * kKs) * 32 + lane};
        uint4 b[2][kGp], bn[2][kGp];
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int p = 0; p < kGp; ++p) b[n][p] = __ldg(wp[n] + 32 * p);
#pragma unroll 1
        for (int gr = 0; gr < kGroups; gr += 2)
        {
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int p = 0; p < kGp; ++p) bn[n][p] = __ldg(wp[n] + 32 * (kGp + p));
            proj_group<kK, kGp, 0>(acc, arow, sa, kGroups, gr, gr * kGp, b, lane);
            const int adv = gr + 2 < kGroups ? 64 * kGp : 32 * kGp;  // the last pair re-reads group gr + 1
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                wp[n] += adv;
#pragma unroll
                for (int p = 0; p < kGp; ++p) b[n][p] = __ldg(wp[n] + 32 * p);
            }
            proj_group<kK, kGp, kGp & 1>(acc, arow, sa, kGroups, gr + 1, (gr + 1) * kGp, bn, lane);
        }
#pragma unroll
        for (int n = 0; n < 2; ++n)
        {
            const float2 ws = __ldg(reinterpret_cast<const float2*>(sw + 8 * (tile0 + n) + 2 * t));
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    const float2 xv = unh2(x[m][2 * np + n][i]);
                    x[m][2 * np + n][i] = h2(__fadd_rn(xv.x, __fmul_rn(acc[m][n][2 * i], ws.x)),
                                             __fadd_rn(xv.y, __fmul_rn(acc[m][n][2 * i + 1], ws.y)));
                }
        }
    }
}

// Feed-forward chunk cq's g (this thread's 8 rows x 2 columns, FP16) as INT8, once every warp's row maxima are in (its
// mbarrier; cq counts chunks over the layers): each row's 16 partials, four per lane of the row's quad; warp 0 writes
// the chunk's token scales.
__device__ __forceinline__ void quantize_chunk(int cq, const uint32_t (&gq)[4][2], unsigned char* smem, int warp,
                                               int lane)
{
    const int g = lane >> 2, t = lane & 3, ch = cq % kChunks;
    auto* sc = reinterpret_cast<Scales*>(smem + kOffScale);
    mbar_wait(reinterpret_cast<uint64_t*>(smem + kOffMbar) + (cq % kGBufs), (cq / kGBufs) & 1);
    const auto* red = reinterpret_cast<const __half2*>(smem + kOffRedG) + (cq % kGBufs) * kTok * kWarps / 2;
    unsigned char* gb = smem + kOffG;
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const int row = 16 * m + g + 8 * i;
            const __half2 m2 = __hmax2(red[(row * kWarps + 4 * t) / 2], red[(row * kWarps + 4 * t) / 2 + 1]);
            float mx = fmaxf(__low2float(m2), __high2float(m2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float inv = mx > 0.f ? 127.f / mx : 0.f;
            if (warp == 0 && t == 0) sc->g[row][ch] = __fmul_rn(mx, 1.f / 127.f);
            const float2 f = unh2(gq[m][i]);
            *reinterpret_cast<uint16_t*>(gb + swz(row, kChunk * ch + 8 * warp + 2 * t, kFf)) =
                static_cast<uint16_t>(round_to_byte(__fmul_rn(f.x, inv)) | round_to_byte(__fmul_rn(f.y, inv)) << 8);
        }
}

__global__ void __launch_bounds__(kThreads, 1)
    fused512_forward(__grid_constant__ const F512Params P, const SmallPackedPosition* __restrict__ packed,
                     const float* __restrict__ dense, unsigned char* __restrict__ out)
{
    extern __shared__ __align__(128) unsigned char smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int b = blockIdx.x;
    auto* sc = reinterpret_cast<Scales*>(smem + kOffScale);
    auto* mbar = reinterpret_cast<uint64_t*>(smem + kOffMbar);
    const auto* pair_bias = static_cast<const uint4*>(P.pair_bias);
    if (tid < kGBufs) mbar_init(mbar + tid, kThreads);

    // ---- Input planes into A as INT8 [64][128] (100 · plane, token scale 1/100), swizzled rows of 128 bytes.
    unsigned char* a = smem + kOffA;
    if (packed)
    {
        auto* pk = reinterpret_cast<unsigned long long*>(smem + kOffB);
        const auto* src = reinterpret_cast<const unsigned long long*>(packed + b);
        for (int i = tid; i < static_cast<int>(sizeof(SmallPackedPosition) / 8); i += kThreads) pk[i] = src[i];
        __syncthreads();
        if (tid < 4 * kTok)  // four threads per token (adjacent lanes): planes 32q..32q + 31
        {
            const auto& pos = *reinterpret_cast<const SmallPackedPosition*>(pk);
            const int cell = tid >> 2, q = tid & 3;
            const unsigned word = plane_word(pos, cell, q, P.stm_colour);
            uint32_t o[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = ((((word >> (4 * j)) & 0xFu) * 0x00204081u) & 0x01010101u) * 100u;
            if (q == 3)  // plane 110: the halfmove clock (at most 100), byte 14 of the thread's 32
                o[3] |= static_cast<uint32_t>(min(static_cast<int>(pos.halfmove), 100)) << 16;
            *reinterpret_cast<uint4*>(a + swz(cell, 32 * q, 128)) = make_uint4(o[0], o[1], o[2], o[3]);
            *reinterpret_cast<uint4*>(a + swz(cell, 32 * q + 16, 128)) = make_uint4(o[4], o[5], o[6], o[7]);
        }
    }
    else
    {
        const float* src = dense + static_cast<size_t>(b) * kTok * kPackPlanes;
        for (int i = tid; i < kTok * 128; i += kThreads)
        {
            const int tok = i >> 7, pl = i & 127;
            const float v = pl < kPackPlanes ? src[tok * kPackPlanes + pl] : 0.f;
            a[swz(tok, pl, 128)] = static_cast<unsigned char>(static_cast<int8_t>(__float2int_rn(v * 100.f)));
        }
    }
    __syncthreads();

    // ---- Embedding: x0 = planes · We + be + pos, FP16; the residual stream starts from it (registers) and the copy
    // re-injected into every layer goes to the global scratch.
    uint32_t x[4][4][2];
#pragma unroll
    for (int np = 0; np < 2; ++np)
    {
        const int tile[2] = {4 * warp + 2 * np, 4 * warp + 2 * np + 1};
        uint4 bw[2];
        load_b<2>(bw, static_cast<const uint4*>(P.embed_w), 2, tile, lane);
        int c[4][2][4];
        gemm64<2>(c, a, 128, static_cast<const uint4*>(P.embed_w), 2, tile, lane, bw);
#pragma unroll
        for (int n = 0; n < 2; ++n)
        {
            const int col = 8 * tile[n] + 2 * t;
            const float2 ws = __ldg(reinterpret_cast<const float2*>(P.embed_s + col));
            const float2 eb = __ldg(reinterpret_cast<const float2*>(P.embed_b + col));
            const float s0 = __fmul_rn(0.01f, ws.x), s1 = __fmul_rn(0.01f, ws.y);
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    const int row = 16 * m + g + 8 * i;
                    const float2 pe = __ldg(reinterpret_cast<const float2*>(P.pos + row * kD + col));
                    x[m][2 * np + n][i] = h2(__int2float_rn(c[m][n][2 * i]) * s0 + eb.x + pe.x,
                                             __int2float_rn(c[m][n][2 * i + 1]) * s1 + eb.y + pe.y);
                }
        }
    }
    uint4* const x0s = x0_slot(P.x0_scratch, warp, lane);
    {
        const auto* xw = reinterpret_cast<const uint4*>(&x[0][0][0]);
#pragma unroll
        for (int k = 0; k < 8; ++k) x0s[32 * k] = xw[k];
    }

    // ---- Layers
#pragma unroll 1
    for (int l = 0; l < P.layers; ++l)
    {
        const F512Layer& L = P.layer[l];
        if (tid < kHd) sc->kfold[tid] = __ldg(L.kfold + tid);  // read after the norm's barriers
        norm_i8<true, false>(x, x0s, L.gain, L.norm1, smem, warp, lane, tid);

        if (warp < kQkvWarps)
        {
            // Producers: this warp's 2 n-tiles of each head (q 0..7, k 8..15, v 16..23 of the head's 24).
            const auto* wq = static_cast<const uint4*>(L.qkv);
            auto tile_of = [warp](int h, int n) {
                const int ht = 2 * warp + n;
                return (ht >> 3) * (kD / 8) + 8 * h + (ht & 7);
            };
            const int which = warp >> 2;  // q, k, v
            uint4 bq[2];
            {
                const int t0[2] = {tile_of(0, 0), tile_of(0, 1)};
                load_b<2>(bq, wq, kD / 64, t0, lane);
            }
            float2 kf[2];
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                const int dim = 8 * ((2 * warp + n) & 7) + 2 * t;
                kf[n] = which == 1 ? *reinterpret_cast<const float2*>(sc->kfold + dim) : make_float2(1.f, 1.f);
            }
#pragma unroll 1
            for (int h = 0; h < kHeads; ++h)
            {
                const int tile[2] = {tile_of(h, 0), tile_of(h, 1)};
                int c[4][2][4];
                gemm64<2>(c, smem + kOffA, kD, wq, kD / 64, tile, lane, bq);
                if (h + 1 < kHeads)
                {
                    const int tn[2] = {tile_of(h + 1, 0), tile_of(h + 1, 1)};
                    load_b<2>(bq, wq, kD / 64, tn, lane);
                }
                float2 ws[2];
#pragma unroll
                for (int n = 0; n < 2; ++n) ws[n] = __ldg(reinterpret_cast<const float2*>(L.qkv_s + 8 * tile[n] + 2 * t));
                if (h > 0) bar_sync(kBarQkvFree);  // the attention warps are done with head h - 1
                unsigned char* qkv = smem + kOffQkv + which * kTok * kHd * 2;
                auto* part = reinterpret_cast<float*>(smem + kOffQkPart);
#pragma unroll
                for (int m = 0; m < 4; ++m)
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                    {
                        const int row = 16 * m + g + 8 * i;
                        const float ra = sc->a[row];
                        float ss = 0.f;
#pragma unroll
                        for (int n = 0; n < 2; ++n)
                        {
                            const int dim = 8 * ((2 * warp + n) & 7) + 2 * t;
                            const float v0 = __int2float_rn(c[m][n][2 * i]) * __fmul_rn(ra, ws[n].x);
                            const float v1 = __int2float_rn(c[m][n][2 * i + 1]) * __fmul_rn(ra, ws[n].y);
                            ss = __fmaf_rn(v0, v0, __fmaf_rn(v1, v1, ss));
                            *reinterpret_cast<uint32_t*>(qkv + swz(row, 2 * dim, 128)) = h2(v0 * kf[n].x, v1 * kf[n].y);
                        }
                        if (which < 2)
                        {
                            ss += __shfl_xor_sync(0xffffffffu, ss, 1);
                            ss += __shfl_xor_sync(0xffffffffu, ss, 2);
                            if (t == 0) part[(which * kTok + row) * 4 + (warp & 3)] = ss;
                        }
                    }
                bar_arrive(kBarQkvFull);
            }
        }
        else
        {
#pragma unroll 1
            for (int h = 0; h < kHeads; ++h)
            {
                bar_sync(kBarQkvFull);
                attention_head(h, warp - kQkvWarps, pair_bias, smem, lane);
                if (h + 1 < kHeads) bar_arrive(kBarQkvFree);
            }
        }
        __syncthreads();  // o complete
        project_into<kD, 1>(x, smem + kOffO, &sc->o[0][0], static_cast<const uint4*>(L.out), L.out_s, warp, lane);
        __syncthreads();  // o's reads before the norm's partials (region B)

        // ---- Feed-forward
        norm_i8<false, false>(x, nullptr, 0.f, L.norm2, smem, warp, lane, tid);
        const auto* wu = static_cast<const uint4*>(L.up);
        uint4 bu[2];
        uint32_t gp[4][2];  // the previous chunk's g, quantized after this chunk's multiply
        {
            const int t0[2] = {2 * warp, 2 * warp + 1};
            load_b<2>(bu, wu, kD / 64, t0, lane);
        }
#pragma unroll 1
        for (int ch = 0; ch < kChunks; ++ch)
        {
            const int cq = l * kChunks + ch;
            const int gt = (kChunk / 8) * ch + warp;  // g n-tile: columns 8gt..8gt+7
            const int tile[2] = {2 * gt, 2 * gt + 1};
            int c[4][2][4];
            gemm64<2>(c, smem + kOffA, kD, wu, kD / 64, tile, lane, bu);
            if (ch + 1 < kChunks)
            {
                const int tn[2] = {tile[0] + kChunk / 4, tile[1] + kChunk / 4};
                load_b<2>(bu, wu, kD / 64, tn, lane);
            }
            const float2 wg = __ldg(reinterpret_cast<const float2*>(L.up_s + 8 * tile[0] + 2 * t));
            const float2 wv = __ldg(reinterpret_cast<const float2*>(L.up_s + 8 * tile[1] + 2 * t));
            auto* red = reinterpret_cast<__half*>(smem + kOffRedG) + (cq % kGBufs) * kTok * kWarps;
            uint32_t gv[4][2];
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                {
                    const int row = 16 * m + g + 8 * i;
                    const float ra = sc->a[row];
                    const float ga = __int2float_rn(c[m][0][2 * i]) * __fmul_rn(ra, wg.x);
                    const float gb = __int2float_rn(c[m][0][2 * i + 1]) * __fmul_rn(ra, wg.y);
                    const float ua = __int2float_rn(c[m][1][2 * i]) * __fmul_rn(ra, wv.x);
                    const float ub = __int2float_rn(c[m][1][2 * i + 1]) * __fmul_rn(ra, wv.y);
                    gv[m][i] = h2(silu_fast(ga) * ua, silu_fast(gb) * ub);
                    const __half2 a2 = __habs2(*reinterpret_cast<const __half2*>(&gv[m][i]));
                    float mx = fmaxf(__low2float(a2), __high2float(a2));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                    if (t == 0) red[row * kWarps + warp] = __float2half_rn(mx);  // exact: an FP16 value
                }
            mbar_arrive(mbar + (cq % kGBufs));
            if (ch > 0) quantize_chunk(cq - 1, gp, smem, warp, lane);
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int i = 0; i < 2; ++i) gp[m][i] = gv[m][i];
        }
        quantize_chunk(l * kChunks + kChunks - 1, gp, smem, warp, lane);
        __syncthreads();  // g complete
        project_into<kFf, 2>(x, smem + kOffG, &sc->g[0][0], static_cast<const uint4*>(L.down), L.down_s, warp, lane);
        __syncthreads();  // g's reads before the next norm's partials (region B)
    }

    // ---- Final norm into A, and the token mean of its FP32 output (the value heads' input).
    norm_i8<false, true>(x, nullptr, 0.f, P.norm, smem, warp, lane, tid);

    // ---- Policy: q (pre-scaled by 1/sqrt(d) in the file) and k in passes of 128 columns through shared memory;
    // ft[from][to] = q_from · k_to and pt[promotion][to] = promo · k_to accumulate over the passes. Warp w computes
    // from-tile w & 3 for to-tiles 2(w >> 2), + 1, and warps 0-7 the promotions for to-tile w.
    float ft[2][4] = {}, pt[4] = {};
    const auto* wpol = static_cast<const uint4*>(P.pol_w);
    unsigned char* qc = smem + kOffPolQ;
    unsigned char* kc = smem + kOffPolK;
#pragma unroll 1
    for (int pass = 0; pass < kD / kPolChunk; ++pass)
    {
        {
            const int half = warp >> 3;  // q, k
            const int tile[2] = {half * (kD / 8) + (kPolChunk / 8) * pass + 2 * (warp & 7),
                                 half * (kD / 8) + (kPolChunk / 8) * pass + 2 * (warp & 7) + 1};
            uint4 bp[2];
            load_b<2>(bp, wpol, kD / 64, tile, lane);
            int c[4][2][4];
            gemm64<2>(c, smem + kOffA, kD, wpol, kD / 64, tile, lane, bp);
            unsigned char* dst = half ? kc : qc;
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                const float2 ws = __ldg(reinterpret_cast<const float2*>(P.pol_s + 8 * tile[n] + 2 * t));
                const float2 bb = __ldg(reinterpret_cast<const float2*>(P.pol_b + 8 * tile[n] + 2 * t));
                const int col = 16 * (warp & 7) + 8 * n + 2 * t;  // within the pass
#pragma unroll
                for (int m = 0; m < 4; ++m)
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                    {
                        const int row = 16 * m + g + 8 * i;
                        const float ra = sc->a[row];
                        *reinterpret_cast<uint32_t*>(dst + swz(row, 2 * col, 2 * kPolChunk)) =
                            h2(__int2float_rn(c[m][n][2 * i]) * __fmul_rn(ra, ws.x) + bb.x,
                               __int2float_rn(c[m][n][2 * i + 1]) * __fmul_rn(ra, ws.y) + bb.y);
                    }
            }
        }
        __syncthreads();
        {
            const int mt = warp & 3, nt0 = 2 * (warp >> 2);
#pragma unroll
            for (int kk = 0; kk < kPolChunk / 16; ++kk)
            {
                uint32_t af[4], bf[4];
                ldsm4(af, qc + swz(16 * mt + (lane & 15), 32 * kk + 16 * (lane >> 4), 2 * kPolChunk));
                ldsm4(bf, kc + swz(8 * nt0 + (lane & 7) + 8 * (lane >> 4), 32 * kk + 16 * ((lane >> 3) & 1),
                                   2 * kPolChunk));
                mma(ft[0], af, bf[0], bf[1]);
                mma(ft[1], af, bf[2], bf[3]);
            }
            if (warp < 8)
            {
                const auto* promo = static_cast<const uint32_t*>(P.promo);  // [4][d] FP16 pairs
#pragma unroll
                for (int kk = 0; kk < kPolChunk / 16; kk += 2)
                {
                    uint32_t bf[4];  // k-steps kk and kk + 1 of to-tile w
                    ldsm4(bf, kc + swz(8 * warp + (lane & 7), 32 * kk + 16 * (lane >> 3), 2 * kPolChunk));
#pragma unroll
                    for (int j = 0; j < 2; ++j)
                    {
                        const int k0 = kPolChunk * pass + 16 * (kk + j);
                        const uint32_t pa[4] = {g < 4 ? __ldg(promo + (g * kD + k0) / 2 + t) : 0u, 0u,
                                                g < 4 ? __ldg(promo + (g * kD + k0 + 8) / 2 + t) : 0u, 0u};
                        mma(pt, pa, bf[2 * j], bf[2 * j + 1]);
                    }
                }
            }
        }
        __syncthreads();  // before the next pass overwrites q and k
    }
    float* scores = reinterpret_cast<float*>(smem + kOffScores);  // [68][kLdS]: from-token rows, then promotions
    {
        const int mt = warp & 3, nt0 = 2 * (warp >> 2);
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                *reinterpret_cast<float2*>(scores + (16 * mt + g + 8 * i) * kLdS + 8 * (nt0 + n) + 2 * t) =
                    make_float2(ft[n][2 * i], ft[n][2 * i + 1]);
        if (warp < 8 && g < 4)
            *reinterpret_cast<float2*>(scores + (kTok + g) * kLdS + 8 * warp + 2 * t) = make_float2(pt[0], pt[1]);
    }
    __syncthreads();
    auto* policy = reinterpret_cast<__half*>(out + static_cast<size_t>(b) * kRecordBytes);
    for (int i = tid; i < kPolicySize; i += kThreads)
    {
        const unsigned code = __ldg(P.vocab + i);
        const int to = (code >> 6) & 63;
        policy[i] = __float2half(scores[(code & 63) * kLdS + to] + scores[(kTok + ((code >> 12) & 3)) * kLdS + to]);
    }

    // ---- Value heads: MLPs on the pooled vector (first layers FP16 [mlp][in][out]: a thread takes two adjacent
    // units over all inputs, the warp's loads coalesced), second layers one warp per output.
    const auto* pooled = reinterpret_cast<const float*>(smem + kOffPooled);
    auto* hidden = reinterpret_cast<float*>(smem + kOffHidden);
    const int n_mlp = P.n_wdl + 1 + P.mlhq_from_q;
    for (int item = tid; item < n_mlp * kD / 2; item += kThreads)
    {
        const int mlp = item / (kD / 2), u = 2 * (item % (kD / 2));
        const auto* w = reinterpret_cast<const __half2*>(static_cast<const __half*>(P.mlp1_w) +
                                                         static_cast<size_t>(mlp) * kD * kD + u);
        float a0 = 0.f, a1 = 0.f;
#pragma unroll 8
        for (int k = 0; k < kD; ++k)
        {
            const float2 wv = __half22float2(__ldg(w + k * (kD / 2)));
            const float pk = pooled[k];
            a0 = fmaf(wv.x, pk, a0);
            a1 = fmaf(wv.y, pk, a1);
        }
        const float b0 = __ldg(P.mlp1_b + mlp * kD + u), b1 = __ldg(P.mlp1_b + mlp * kD + u + 1);
        hidden[mlp * kD + u] = (a0 + b0) / (1.f + __expf(-(a0 + b0)));
        hidden[mlp * kD + u + 1] = (a1 + b1) / (1.f + __expf(-(a1 + b1)));
    }
    __syncthreads();
    const int n_out = 3 * P.n_wdl + 1 + P.mlhq_from_q;
    for (int o = warp; o < n_out; o += kWarps)
    {
        const int mlp = o < 3 * P.n_wdl ? o / 3 : P.n_wdl + (o - 3 * P.n_wdl);
        float acc = 0.f;
        for (int k = lane; k < kD; k += 32) acc += __ldg(P.mlp2_w + o * kD + k) * hidden[mlp * kD + k];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) sc->heads[o] = acc + __ldg(P.mlp2_b + o);
    }
    __syncthreads();
    if (tid == 0)
    {
        float wdl[3] = {0.f, 0.f, 0.f};
        for (int k = 0; k < P.n_wdl; ++k)
        {
            const float* z = sc->heads + 3 * k;
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
        aux[3] = sc->heads[3 * P.n_wdl];
        aux[4] = P.mlhq_from_q ? sc->heads[3 * P.n_wdl + 1] : aux[3];
    }
}

} // namespace

cudaError_t launch_fused512_forward(const F512Params& params, const SmallPackedPosition* packed, const float* dense,
                                    int batch, void* out, cudaStream_t stream)
{
    if (batch <= 0) return cudaSuccess;
    static const cudaError_t setup = [] {
        const cudaError_t e = cudaFuncSetAttribute(fused512_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        if (e != cudaSuccess) return e;
        return cudaFuncSetAttribute(fused512_forward, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared);
    }();
    if (setup != cudaSuccess) return setup;
    fused512_forward<<<batch, kThreads, kSmem, stream>>>(params, packed, dense, static_cast<unsigned char*>(out));
    return cudaGetLastError();
}

} // namespace chessrl::net
