#pragma once

// Encoder networks wider than the fused kernel holds (src/export/to_cnet.py entries_for_wide: d_model and d_ff in
// multiples of 64 up to 1024, heads of 16 to 64 columns). Their projections run as cuBLAS FP16 multiplies with FP32
// sums over the whole batch's tokens, the rest as small kernels (encoder_wide.cu); the output records are the fused
// kernel's (encoder_params.hpp). Plain C++ so the g++ loader and nvcc share it.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "mcts/packed_position.hpp"
#include "net/encoder_params.hpp"

namespace chessrl::net
{

constexpr int kWideMaxD = 1024;

// Device pointers into a wide .cnet; matrices are FP16 row-major [out][in].
struct WideLayer
{
    const float* norm1;
    const void* qkv;     // [3d][d]
    const float* qnorm;  // [head_dim], times log2(e) / sqrt(head_dim)
    const float* knorm;  // [head_dim]
    const void* out;     // [d][d]
    const float* norm2;
    const void* up;      // [2 d_ff][d]: gate rows, then up rows
    const void* down;    // [d][d_ff]
    float gain;          // x0_gain
};

struct WideParams
{
    int d;
    int layers;
    int heads;
    int d_ff;
    int n_wdl;
    float wdl_weight[2];
    int mlhq_from_q;
    int stm_colour;
    const void* embed_w;    // [d][128], planes padded to 128
    const float* embed_b;   // [d]
    const float* pos;       // [64][d]
    const void* pair_bias;  // [heads][64][64], times log2(e)
    WideLayer layer[kMaxLayers];
    const float* norm;
    const void* pol_w;  // [2d][d]: pol_q (times 1/sqrt(d)), then pol_k
    const float* pol_b; // [2d]
    const void* promo;  // [4][d], times 1/sqrt(d)
    const std::uint16_t* vocab;
    const void* mlp1_w;  // [n_mlp][d][d]
    const float* mlp1_b;
    const float* mlp2_w;  // [n_out][d]
    const float* mlp2_b;
};

// Bytes of the per-stream scratch a batch needs.
std::size_t wide_scratch_bytes(const WideParams& p, int batch);

// A cuBLAS handle bound to `stream` (one per stream: the self-play workers call concurrently), and its release.
cudaError_t wide_create_handle(void** handle, cudaStream_t stream);
void wide_destroy_handle(void* handle);

// Runs `batch` positions on `stream` (whose handle and scratch these are), one kRecordBytes record each into `out`.
// Exactly one of `packed` and `dense` ([batch][64][111] floats) is non-null.
cudaError_t launch_wide_forward(const WideParams& p, void* handle, void* scratch,
                                const mcts::single::SmallPackedPosition* packed, const float* dense, int batch,
                                void* out, cudaStream_t stream);

} // namespace chessrl::net
