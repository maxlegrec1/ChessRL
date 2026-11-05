#pragma once

// INT8 encoder networks of d_model 512, 8 heads of 64 and d_ff 768 (src/export/to_cnet.py entries_for_fused512, a
// "fused512" entry in the file): one kernel launch runs the whole network, one 16-warp thread block per position
// (encoder_fused512.cu). The output records are the fused kernel's (encoder_params.hpp). Plain C++ so the g++ loader
// and nvcc share it.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "mcts/packed_position.hpp"
#include "net/encoder_params.hpp"

namespace chessrl::net
{

constexpr int kF512D = 512;
constexpr int kF512Heads = 8;
constexpr int kF512Ff = 768;
// Per position, the kernel's global scratch: x0, the FP16 embedding re-injected into every layer.
constexpr std::size_t kF512X0Bytes = 64 * 512 * 2;

// The projections are INT8 B fragments (to_cnet.pack_b8) with one FP32 scale per output row (*_s).
struct F512Layer
{
    const float* norm1;
    const void* qkv;    // [3d][d]: q, k, v rows
    const float* qkv_s;
    const float* kfold;  // [64]: qnorm · knorm (times log2(e) / 8), the QK-norm's weights folded onto k
    const void* out;
    const float* out_s;
    const float* norm2;
    const void* up;  // [2 d_ff][d], rows interleaved per 8: gate columns 8j..8j+7, then up columns 8j..8j+7
    const float* up_s;
    const void* down;
    const float* down_s;
    float gain;  // x0_gain
};

struct F512Params
{
    int layers;
    int n_wdl;
    float wdl_weight[2];
    int mlhq_from_q;
    int stm_colour;
    const void* embed_w;  // [d][128], planes padded to 128
    const float* embed_s;
    const float* embed_b;
    const float* pos;        // [64][d]
    const void* pair_bias;   // to_cnet.pack_pair_bias, times log2(e)
    F512Layer layer[kMaxLayers];
    const float* norm;
    const void* pol_w;  // [2d][d]: pol_q (times 1/sqrt(d)), then pol_k
    const float* pol_s;
    const float* pol_b;  // [2d]
    const void* promo;   // FP16 [4][d], times 1/sqrt(d)
    const std::uint16_t* vocab;
    const void* mlp1_w;  // FP16 [mlp][in][out]
    const float* mlp1_b;
    const float* mlp2_w;  // [n_out][d]
    const float* mlp2_b;
    void* x0_scratch;  // batch × kF512X0Bytes, set per launch (EncoderNet::forward)
};

// Runs `batch` positions on `stream`, one kRecordBytes record each into `out`. Exactly one of `packed` and `dense`
// ([batch][64][111] floats) is non-null.
cudaError_t launch_fused512_forward(const F512Params& params, const mcts::single::SmallPackedPosition* packed,
                                    const float* dense, int batch, void* out, cudaStream_t stream);

} // namespace chessrl::net
