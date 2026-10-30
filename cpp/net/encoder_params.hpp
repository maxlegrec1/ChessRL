#pragma once

// Device pointers into a loaded .cnet file (src/export/to_cnet.py), passed by value to the
// fused kernel (encoder_kernel.cu). Plain C++ so the g++ loader and nvcc share it.

#include <cuda_runtime_api.h>

#include <cstdint>

#include "mcts/packed_position.hpp"

namespace chessrl::net
{

constexpr int kMaxLayers = 16;
constexpr int kPolicySize = 1929;
// One output record per position: 1929 FP16 policy logits, then (from kAuxOffset) FP32
// WDL probabilities, mlh and mlh_q.
constexpr int kAuxOffset = 3872;
constexpr int kRecordBytes = 3904;

struct LayerParams
{
    const float* norm1;
    const void* qkv;  // packed B fragments, see to_cnet.pack_b
    const float* qnorm;
    const float* knorm;
    const void* out;
    const float* norm2;
    const void* up;
    const void* down;
    float gain;  // x0_gain
};

struct EncoderParams
{
    int layers;
    int n_wdl;        // value MLPs averaged into the WDL (2 for search_heads=blend)
    int mlhq_from_q;  // mlh_q output from the mlh_q head (search) or repeating mlh
    const void* embed_w;
    const float* embed_b;
    const float* pos;
    const void* pair_bias;
    LayerParams layer[kMaxLayers];
    const float* norm;
    const void* pol_w;
    const float* pol_b;
    const void* promo;
    float special;
    const std::uint16_t* vocab;
    const void* mlp1_w;
    const float* mlp1_b;
    const float* mlp2_w;
    const float* mlp2_b;
};

// Runs `batch` positions on `stream`, one kRecordBytes record each into `out`. Exactly
// one of `packed` and `dense` ([batch][64][111] floats, the dense planes) is non-null.
cudaError_t launch_encoder_forward(const EncoderParams& params,
                                   const mcts::single::SmallPackedPosition* packed,
                                   const float* dense,
                                   int batch,
                                   void* out,
                                   cudaStream_t stream);

} // namespace chessrl::net
