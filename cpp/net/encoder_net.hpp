#pragma once

// The fused network: a .cnet file (src/export/to_cnet.py) loaded onto the GPU, run by
// encoder_kernel.cu on a caller's stream.

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

#include "mcts/packed_position.hpp"
#include "net/encoder_params.hpp"

namespace chessrl::net
{

class EncoderNet
{
public:
    explicit EncoderNet(std::string path);
    ~EncoderNet();
    EncoderNet(const EncoderNet&) = delete;
    EncoderNet& operator=(const EncoderNet&) = delete;

    // `batch` output records of kRecordBytes into d_out; one of d_packed / d_dense
    // ([batch][64][111] floats) is non-null. Asynchronous on `stream`.
    cudaError_t forward(const mcts::single::SmallPackedPosition* d_packed, const float* d_dense, int batch,
                        void* d_out, cudaStream_t stream) const;

    // Synchronous helpers for tests and tools: host input in, host records out.
    std::vector<unsigned char> run_dense(const float* planes, int batch) const;
    std::vector<unsigned char> run_packed(const mcts::single::SmallPackedPosition* positions, int batch) const;

    const std::string& path() const { return path_; }
    int layers() const { return params_.layers; }

private:
    std::vector<unsigned char> run_host(const void* input, size_t bytes, bool packed, int batch) const;

    std::string path_;
    void* blob_{nullptr};
    EncoderParams params_{};
};

// True for paths the fused network loads (the .cnet extension).
bool is_cnet_path(const std::string& path);

// Kernel throughput in positions per second: `streams` streams, each running `iterations`
// batches of `batch` copies of the start position back to back (the output copy included).
double benchmark(const EncoderNet& net, int batch, int iterations, int streams);

} // namespace chessrl::net
