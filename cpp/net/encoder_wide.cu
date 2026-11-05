// Forward pass of encoder networks wider than the fused kernel holds (src/models/encoder.py; the .cnet from
// src/export/to_cnet.py entries_for_wide). At d_model 512 one position's activations no longer fit a multiprocessor's
// shared memory, but the projections are large enough for cuBLAS: every projection is one FP16 multiply over all the
// batch's tokens (64 per position), with FP16 sums as the trainer's multiplies (fp16_accumulation; CHESSRL_FP16_ACC=0:
// FP32 sums). The residual stream is FP16, as in training (fp16_residual): the out and down projections add into it
// as the multiply's beta = 1. Attention is cuBLAS too, per head over the batch's positions: QK-norm in place, scores
// Q K^T in FP32, the pair bias and an exp2 softmax that writes P in FP16 over its own score rows, then P V. The rest
// are small kernels: the input planes (net/planes.cuh, as the fused kernel), the norms (one warp per token, in FP32),
// SwiGLU, the policy logits and the value heads (from the final norm's FP32 output). Output records as the fused
// kernel's (encoder_params.hpp).

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "net/encoder_wide.hpp"
#include "net/planes.cuh"

namespace chessrl::net
{
namespace
{

using mcts::single::kPackPlanes;
using mcts::single::SmallPackedPosition;

constexpr int kTok = 64;
constexpr int kPlanesLd = 128;                   // input planes padded to 128 (the embedding's columns)
constexpr int kScores = kTok * kTok + 4 * kTok;  // per position: from x to scores, then promotion x to
constexpr float kEps = 1e-6f;
constexpr int kMaxMlps = 4;

// FP16 sums in the projections unless CHESSRL_FP16_ACC=0 (the fused kernel's setting; there N terms per FP16 sum).
bool fp16_sums()
{
    static const bool on = [] {
        const char* env = std::getenv("CHESSRL_FP16_ACC");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return on;
}

struct Layout
{
    std::size_t planes, x0, h, a, big, mid, s, sc, pooled, hidden, total;
};

Layout layout(const WideParams& p, int batch)
{
    const std::size_t n = static_cast<std::size_t>(batch) * kTok;
    const std::size_t d = static_cast<std::size_t>(p.d), ff = static_cast<std::size_t>(p.d_ff);
    std::size_t off = 0;
    const auto take = [&off](std::size_t bytes) {
        const std::size_t at = off;
        off += (bytes + 255) & ~static_cast<std::size_t>(255);
        return at;
    };
    Layout l{};
    l.planes = take(n * kPlanesLd * 2);
    l.x0 = take(n * d * 2);  // the FP16 embedding re-injected into every layer
    l.h = take(n * d * 2);   // the FP16 residual stream
    l.a = take(n * d * 2);   // a norm's FP16 output: the next multiply's input
    // qkv, the up projection, the final norm's FP32 output (before the policy's q and k overwrite it)
    l.big = take(n * std::max({3 * d, 2 * ff, 2 * d}) * 2);
    l.mid = take(n * std::max(d, ff) * 2);  // the attention output, the SwiGLU output
    l.s = take(static_cast<std::size_t>(batch) * p.heads * kTok * kTok * 4);  // attention scores, then P over them
    l.sc = take(static_cast<std::size_t>(batch) * kScores * 4);
    l.pooled = take(static_cast<std::size_t>(batch) * d * 2);
    l.hidden = take(static_cast<std::size_t>(batch) * kMaxMlps * d * 4);
    l.total = off;
    return l;
}

int blocks(std::size_t n, int threads) { return static_cast<int>((n + threads - 1) / threads); }

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

__global__ void expand_packed(const SmallPackedPosition* packed, int stm_colour, __half* planes)
{
    // One position per block, four threads per token (planes_to_row's layout).
    const int b = blockIdx.x, tid = threadIdx.x;
    planes_to_row(packed[b], tid >> 2, tid & 3, stm_colour,
                  planes + (static_cast<std::size_t>(b) * kTok + (tid >> 2)) * kPlanesLd);
}

__global__ void expand_dense(const float* dense, __half* planes, std::size_t n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::size_t tok = i / kPlanesLd;
    const int pl = static_cast<int>(i % kPlanesLd);
    planes[i] = __float2half(pl < kPackPlanes ? dense[tok * kPackPlanes + pl] : 0.f);
}

// h = planes . embed^T (from the multiply) + bias + pos, in FP32, rounded once; x0 = h.
__global__ void embed_finish(__half* h, __half* x0, const float* bias, const float* pos, int d, std::size_t n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::size_t tok = i / d;
    const int c = static_cast<int>(i % d);
    const __half v = __float2half(__half2float(h[i]) + bias[c] + pos[(tok % kTok) * d + c]);
    h[i] = v;
    x0[i] = v;
}

// One warp per token, kP2 = d / 64 half2 pairs per lane (registers): with kX0, h += gain * x0 first in FP32, rounded
// once and written back (the residual continues from there); then the RMS norm in FP32 into out16, and out32 when
// given.
template <bool kX0, int kP2>
__global__ void rmsnorm(__half* h, const __half* x0, float gain, const float* w, __half* out16, float* out32,
                        int n_tok)
{
    constexpr int kD = 64 * kP2;
    const int tok = static_cast<int>((static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (tok >= n_tok) return;
    auto* row = reinterpret_cast<__half2*>(h + static_cast<std::size_t>(tok) * kD);
    float2 v[kP2];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kP2; ++i)
    {
        const int j = lane + 32 * i;
        float2 x = __half22float2(row[j]);
        if (kX0)
        {
            const float2 z = __half22float2(reinterpret_cast<const __half2*>(x0 + static_cast<std::size_t>(tok) * kD)[j]);
            const __half2 r = __floats2half2_rn(x.x + gain * z.x, x.y + gain * z.y);
            row[j] = r;
            x = __half22float2(r);
        }
        v[i] = x;
        ss += x.x * x.x + x.y * x.y;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float inv = rsqrtf(ss / static_cast<float>(kD) + kEps);
    const auto* w2 = reinterpret_cast<const float2*>(w);
    auto* o16 = reinterpret_cast<__half2*>(out16 + static_cast<std::size_t>(tok) * kD);
    auto* o32 = out32 ? reinterpret_cast<float2*>(out32 + static_cast<std::size_t>(tok) * kD) : nullptr;
#pragma unroll
    for (int i = 0; i < kP2; ++i)
    {
        const int j = lane + 32 * i;
        const float2 wv = w2[j];
        const float2 y = make_float2(v[i].x * inv * wv.x, v[i].y * inv * wv.y);
        o16[j] = __floats2half2_rn(y.x, y.y);
        if (o32) o32[j] = y;
    }
}

// rmsnorm for d = 64 * p2 (p2 up to kWideMaxD / 64), one warp per token.
template <bool kX0, int kP2 = 1>
void launch_rmsnorm(int p2, __half* h, const __half* x0, float gain, const float* w, __half* out16, float* out32,
                    int n_tok, cudaStream_t stream)
{
    if constexpr (kP2 <= kWideMaxD / 64)
    {
        if (p2 == kP2)
        {
            const int threads = 256;
            const int grid = static_cast<int>((static_cast<std::size_t>(n_tok) * 32 + threads - 1) / threads);
            rmsnorm<kX0, kP2><<<grid, threads, 0, stream>>>(h, x0, gain, w, out16, out32, n_tok);
        }
        else
        {
            launch_rmsnorm<kX0, kP2 + 1>(p2, h, x0, gain, w, out16, out32, n_tok, stream);
        }
    }
}

// QK-norm in place: one warp per (token, head), the head's q and k rows of qkv [tokens][3d] RMS-normalized in FP32 as
// half2 pairs, one per lane (hd up to 64; qnorm carries log2(e) / sqrt(hd), for the exp2 softmax).
__global__ void qk_norm(__half* qkv, const float* qnorm, const float* knorm, int d, int heads, int hd, int n_tok)
{
    const int item = static_cast<int>((static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (item >= n_tok * heads) return;
    const int tok = item / heads, head = item % heads;
    const bool live = 2 * lane < hd;
#pragma unroll
    for (int part = 0; part < 2; ++part)
    {
        auto* row = reinterpret_cast<__half2*>(qkv + static_cast<std::size_t>(tok) * 3 * d + part * d + head * hd);
        const float2 v = live ? __half22float2(row[lane]) : make_float2(0.f, 0.f);
        float ss = v.x * v.x + v.y * v.y;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
        const float inv = rsqrtf(ss / static_cast<float>(hd) + kEps);
        if (live)
        {
            const float2 w = reinterpret_cast<const float2*>(part == 0 ? qnorm : knorm)[lane];
            row[lane] = __floats2half2_rn(v.x * inv * w.x, v.y * inv * w.y);
        }
    }
}

// One warp per score row (position, head, query): + the pair bias (times log2(e)), softmax in exp2, P written in FP16
// over the start of its own FP32 row (row stride 128 halves), every lane's reads done before any write.
__global__ void softmax_rows(float* s, const __half* bias, int heads, std::size_t n_rows)
{
    const std::size_t row = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= n_rows) return;
    const int q = static_cast<int>(row % kTok), head = static_cast<int>((row / kTok) % heads);
    float* r = s + row * kTok;
    const __half* br = bias + (static_cast<std::size_t>(head) * kTok + q) * kTok;
    const float x0 = r[lane] + __half2float(br[lane]);
    const float x1 = r[lane + 32] + __half2float(br[lane + 32]);
    float mx = fmaxf(x0, x1);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    const float e0 = exp2f(x0 - mx), e1 = exp2f(x1 - mx);
    float sum = e0 + e1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    const float inv = 1.f / sum;
    auto* p = reinterpret_cast<__half*>(r);
    p[lane] = __float2half(e0 * inv);
    p[lane + 32] = __float2half(e1 * inv);
}

// g = silu(gate) * up, from up [tokens][2 ff] (gate columns first), as half2 pairs: n = tokens * ff / 2.
__global__ void swiglu(const __half* up, __half* g, int ff, std::size_t n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int half_ff = ff / 2;
    const std::size_t tok = i / half_ff;
    const int j = static_cast<int>(i % half_ff);
    const auto* row = reinterpret_cast<const __half2*>(up + tok * 2 * ff);
    const float2 gate = __half22float2(row[j]), u = __half22float2(row[half_ff + j]);
    reinterpret_cast<__half2*>(g)[i] = __floats2half2_rn(silu(gate.x) * u.x, silu(gate.y) * u.y);
}

__global__ void add_bias(__half* x, const float* bias, int cols, std::size_t n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half(__half2float(x[i]) + bias[i % cols]);
}

// Policy logits: entry (from, to, promotion) = ft[from][to] + pt[promotion][to], FP16 into the record.
__global__ void policy_out(const float* sc, const std::uint16_t* vocab, unsigned char* out)
{
    const int b = blockIdx.x;
    const float* ft = sc + static_cast<std::size_t>(b) * kScores;
    const float* pt = ft + kTok * kTok;
    auto* policy = reinterpret_cast<__half*>(out + static_cast<std::size_t>(b) * kRecordBytes);
    for (int i = threadIdx.x; i < kPolicySize; i += blockDim.x)
    {
        const unsigned code = vocab[i];
        const int to = (code >> 6) & 63;
        policy[i] = __float2half(ft[(code & 63) * kTok + to] + pt[((code >> 12) & 3) * kTok + to]);
    }
}

// The pooled vector: the mean over tokens of the final norm's FP32 output, in FP16 for the value MLPs' first layers.
__global__ void pool(const float* hn, __half* pooled, int d)
{
    const int b = blockIdx.x;
    for (int c = threadIdx.x; c < d; c += blockDim.x)
    {
        float acc = 0.f;
        for (int t = 0; t < kTok; ++t) acc += hn[(static_cast<std::size_t>(b) * kTok + t) * d + c];
        pooled[static_cast<std::size_t>(b) * d + c] = __float2half(acc / kTok);
    }
}

struct ValueArgs
{
    int d, n_wdl, mlhq_from_q;
    float wdl_weight[2];
    const float* mlp1_b;
    const float* mlp2_w;
    const float* mlp2_b;
};

// The value heads' second layers on silu(hidden + b1), then the WDL (softmaxes mixed by wdl_weight), mlh, mlh_q into
// the record, as the fused kernel.
__global__ void value_out(const float* hidden, ValueArgs a, unsigned char* out)
{
    __shared__ float heads[2 * kMaxMlps];
    const int b = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    const int n_mlp = a.n_wdl + 1 + a.mlhq_from_q;
    const int n_out = 3 * a.n_wdl + 1 + a.mlhq_from_q;
    const float* hb = hidden + static_cast<std::size_t>(b) * n_mlp * a.d;
    for (int o = warp; o < n_out; o += warps)
    {
        const int mlp = o < 3 * a.n_wdl ? o / 3 : a.n_wdl + (o - 3 * a.n_wdl);
        float acc = 0.f;
        for (int k = lane; k < a.d; k += 32)
            acc += a.mlp2_w[static_cast<std::size_t>(o) * a.d + k] * silu(hb[mlp * a.d + k] + a.mlp1_b[mlp * a.d + k]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) heads[o] = acc + a.mlp2_b[o];
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float wdl[3] = {0.f, 0.f, 0.f};
        for (int k = 0; k < a.n_wdl; ++k)
        {
            const float* z = heads + 3 * k;
            const float mx = fmaxf(z[0], fmaxf(z[1], z[2]));
            const float e0 = expf(z[0] - mx), e1 = expf(z[1] - mx), e2 = expf(z[2] - mx);
            const float scale = a.wdl_weight[k] / (e0 + e1 + e2);
            wdl[0] += e0 * scale;
            wdl[1] += e1 * scale;
            wdl[2] += e2 * scale;
        }
        auto* aux = reinterpret_cast<float*>(out + static_cast<std::size_t>(b) * kRecordBytes + kAuxOffset);
        aux[0] = wdl[0];
        aux[1] = wdl[1];
        aux[2] = wdl[2];
        aux[3] = heads[3 * a.n_wdl];
        aux[4] = a.mlhq_from_q ? heads[3 * a.n_wdl + 1] : aux[3];
    }
}

// y[n][out] (beta 0: =, 1: +=) = x[n][in] . w[out][in]^T in FP16, sums in FP16 (fp16_sums) or FP32.
cublasStatus_t linear16(cublasHandle_t h, const void* x, const void* w, void* y, std::size_t n, int in, int out,
                        float beta = 0.f)
{
    if (fp16_sums())
    {
        const __half alpha = __float2half(1.f), b = __float2half(beta);
        return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out, static_cast<int>(n), in, &alpha, w, CUDA_R_16F, in, x,
                            CUDA_R_16F, in, &b, y, CUDA_R_16F, out, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    const float alpha = 1.f;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out, static_cast<int>(n), in, &alpha, w, CUDA_R_16F, in, x,
                        CUDA_R_16F, in, &beta, y, CUDA_R_16F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// The same into an FP32 y with FP32 sums (the value MLPs' first layers).
cublasStatus_t linear32(cublasHandle_t h, const void* x, const void* w, float* y, std::size_t n, int in, int out)
{
    const float alpha = 1.f, beta = 0.f;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out, static_cast<int>(n), in, &alpha, w, CUDA_R_16F, in, x,
                        CUDA_R_16F, in, &beta, y, CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

} // namespace

std::size_t wide_scratch_bytes(const WideParams& p, int batch) { return layout(p, batch).total; }

cudaError_t wide_create_handle(void** handle, cudaStream_t stream)
{
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return cudaErrorInitializationError;
    if (cublasSetStream(h, stream) != CUBLAS_STATUS_SUCCESS)
    {
        cublasDestroy(h);
        return cudaErrorInitializationError;
    }
    *handle = h;
    return cudaSuccess;
}

void wide_destroy_handle(void* handle)
{
    if (handle) cublasDestroy(static_cast<cublasHandle_t>(handle));
}

cudaError_t launch_wide_forward(const WideParams& p, void* handle, void* scratch, const SmallPackedPosition* packed,
                                const float* dense, int batch, void* out, cudaStream_t stream)
{
    if (batch <= 0) return cudaSuccess;
    auto* h = static_cast<cublasHandle_t>(handle);
    const Layout l = layout(p, batch);
    auto* base = static_cast<unsigned char*>(scratch);
    auto* planes = reinterpret_cast<__half*>(base + l.planes);
    auto* x0 = reinterpret_cast<__half*>(base + l.x0);
    auto* res = reinterpret_cast<__half*>(base + l.h);
    auto* a = reinterpret_cast<__half*>(base + l.a);
    auto* big = reinterpret_cast<__half*>(base + l.big);
    auto* mid = reinterpret_cast<__half*>(base + l.mid);
    auto* s = reinterpret_cast<float*>(base + l.s);
    auto* sc = reinterpret_cast<float*>(base + l.sc);
    auto* pooled = reinterpret_cast<__half*>(base + l.pooled);
    auto* hidden = reinterpret_cast<float*>(base + l.hidden);
    auto* rec = static_cast<unsigned char*>(out);
    const int d = p.d, ff = p.d_ff, heads = p.heads, hd = p.d / p.heads;
    const std::size_t n = static_cast<std::size_t>(batch) * kTok;
    constexpr int kT = 256;
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
    const auto ok = [&st](cublasStatus_t r) {
        if (st == CUBLAS_STATUS_SUCCESS) st = r;
        return st == CUBLAS_STATUS_SUCCESS;
    };
    const float one = 1.f, zero = 0.f;
    const long long tok3 = static_cast<long long>(kTok) * 3 * d;  // one position's rows of qkv

    if (packed)
        expand_packed<<<batch, 4 * kTok, 0, stream>>>(packed, p.stm_colour, planes);
    else
        expand_dense<<<blocks(n * kPlanesLd, kT), kT, 0, stream>>>(dense, planes, n * kPlanesLd);
    if (!ok(linear16(h, planes, p.embed_w, res, n, kPlanesLd, d))) return cudaErrorUnknown;
    embed_finish<<<blocks(n * d, kT), kT, 0, stream>>>(res, x0, p.embed_b, p.pos, d, n * d);

    const int p2 = d / 64;
    const int n_tok = static_cast<int>(n);
    for (int i = 0; i < p.layers; ++i)
    {
        const WideLayer& L = p.layer[i];
        launch_rmsnorm<true>(p2, res, x0, L.gain, L.norm1, a, nullptr, n_tok, stream);
        if (!ok(linear16(h, a, L.qkv, big, n, d, 3 * d))) return cudaErrorUnknown;
        qk_norm<<<blocks(n * heads * 32, kT), kT, 0, stream>>>(big, L.qnorm, L.knorm, d, heads, hd, static_cast<int>(n));
        for (int head = 0; head < heads; ++head)
        {
            // scores^T [key][query] = K [key][hd] . Q^T [hd][query]: row-major scores [query][key], per position.
            if (!ok(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kTok, kTok, hd, &one, big + d + head * hd,
                                               CUDA_R_16F, 3 * d, tok3, big + head * hd, CUDA_R_16F, 3 * d, tok3, &zero,
                                               s + static_cast<std::size_t>(head) * kTok * kTok, CUDA_R_32F, kTok,
                                               static_cast<long long>(heads) * kTok * kTok, batch, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP)))
                return cudaErrorUnknown;
        }
        softmax_rows<<<blocks(n * heads * 32, kT), kT, 0, stream>>>(s, static_cast<const __half*>(p.pair_bias), heads,
                                                                     n * heads);
        for (int head = 0; head < heads; ++head)
        {
            // out^T [hd][query] = V^T [hd][key] . P^T [key][query]; P rows at a stride of 128 halves.
            if (!ok(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, hd, kTok, kTok, &one, big + 2 * d + head * hd,
                                               CUDA_R_16F, 3 * d, tok3,
                                               reinterpret_cast<const __half*>(s + static_cast<std::size_t>(head) * kTok * kTok),
                                               CUDA_R_16F, 2 * kTok, static_cast<long long>(heads) * kTok * kTok * 2, &zero,
                                               mid + head * hd, CUDA_R_16F, d, static_cast<long long>(kTok) * d, batch,
                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)))
                return cudaErrorUnknown;
        }
        if (!ok(linear16(h, mid, L.out, res, n, d, d, 1.f))) return cudaErrorUnknown;
        launch_rmsnorm<false>(p2, res, nullptr, 0.f, L.norm2, a, nullptr, n_tok, stream);
        if (!ok(linear16(h, a, L.up, big, n, d, 2 * ff))) return cudaErrorUnknown;
        swiglu<<<blocks(n * ff / 2, kT), kT, 0, stream>>>(big, mid, ff, n * ff / 2);
        if (!ok(linear16(h, mid, L.down, res, n, ff, d, 1.f))) return cudaErrorUnknown;
    }
    // The final norm: FP16 for the policy's q and k, FP32 (in big, before q and k take it) for the pooled vector.
    auto* fin = reinterpret_cast<float*>(big);
    launch_rmsnorm<false>(p2, res, nullptr, 0.f, p.norm, a, fin, n_tok, stream);
    pool<<<batch, std::min(d, 1024), 0, stream>>>(fin, pooled, d);

    // Policy: q and k [tokens][2d] (+ bias), then per position from x to = q . k^T and promotion x to = promo . k^T.
    if (!ok(linear16(h, a, p.pol_w, big, n, d, 2 * d))) return cudaErrorUnknown;
    add_bias<<<blocks(n * 2 * d, kT), kT, 0, stream>>>(big, p.pol_b, 2 * d, n * 2 * d);
    {
        const long long stride_tok = static_cast<long long>(kTok) * 2 * d;
        if (!ok(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kTok, kTok, d, &one, big + d, CUDA_R_16F,
                                           2 * d, stride_tok, big, CUDA_R_16F, 2 * d, stride_tok, &zero, sc,
                                           CUDA_R_32F, kTok, kScores, batch, CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP)))
            return cudaErrorUnknown;
        if (!ok(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, kTok, 4, d, &one, big + d, CUDA_R_16F, 2 * d,
                                           stride_tok, p.promo, CUDA_R_16F, d, 0, &zero, sc + kTok * kTok,
                                           CUDA_R_32F, kTok, kScores, batch, CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP)))
            return cudaErrorUnknown;
    }
    policy_out<<<batch, kT, 0, stream>>>(sc, p.vocab, rec);

    // Value heads: the MLPs' first layers as one multiply on the pooled vector, then the second layers and the WDL.
    const int n_mlp = p.n_wdl + 1 + p.mlhq_from_q;
    if (!ok(linear32(h, pooled, p.mlp1_w, hidden, static_cast<std::size_t>(batch), d, n_mlp * d)))
        return cudaErrorUnknown;
    ValueArgs va{d, p.n_wdl, p.mlhq_from_q, {p.wdl_weight[0], p.wdl_weight[1]}, p.mlp1_b, p.mlp2_w, p.mlp2_b};
    value_out<<<batch, kT, 0, stream>>>(hidden, va, rec);
    return cudaGetLastError();
}

} // namespace chessrl::net
