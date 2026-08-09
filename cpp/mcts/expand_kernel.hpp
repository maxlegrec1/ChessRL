#pragma once

// Host-callable launcher for the packed→dense GPU expansion. Declared as
// plain C++ (no CUDA types in the signature beyond cudaStream_t, which is a
// pointer typedef) so the g++ TUs can call it; the definition lives in
// expand_kernel.cu, compiled by nvcc and linked in.

#include <cuda_runtime_api.h>

#include "mcts/packed_position.hpp"

namespace chessrl::mcts::single
{

// Zero `d_out` then expand `batch` packed snapshots into the dense
// [batch, 8, 8, 111] tensor (row-major: b*7104 + (row*8+col)*111 + plane),
// bit-identical to expand_packed_to_dense(). out_is_half: 1 ⇒ __half output
// (the TRT fp16 input), 0 ⇒ float (used by the GPU-vs-CPU parity test).
// Both the memset and the kernel are issued on `stream` so the whole thing
// is capturable inside a CUDA graph. Returns the first CUDA error, or
// cudaSuccess.
cudaError_t launch_expand_planes(
    const SmallPackedPosition* d_packed,
    void* d_out,
    int batch,
    int out_is_half,
    cudaStream_t stream);

} // namespace chessrl::mcts::single
