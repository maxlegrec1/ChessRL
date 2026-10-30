#include "net/encoder_net.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>

namespace chessrl::net
{
namespace
{

struct Entry
{
    std::uint32_t dtype;
    std::uint64_t offset;
    std::uint64_t bytes;
};

using mcts::single::kPackElems;
using mcts::single::kPackPlanes;

constexpr std::uint32_t kF16 = 0, kF32 = 1, kU16 = 2, kI32 = 3;
constexpr std::uint64_t kDtypeBytes[] = {2, 4, 2, 4};

void check(cudaError_t err, const std::string& what)
{
    if (err != cudaSuccess)
        throw std::runtime_error("EncoderNet: " + what + ": " + cudaGetErrorString(err));
}

} // namespace

bool is_cnet_path(const std::string& path)
{
    return path.size() >= 5 && path.compare(path.size() - 5, 5, ".cnet") == 0;
}

EncoderNet::EncoderNet(std::string path)
    : path_(std::move(path))
{
    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("EncoderNet: cannot open " + path_);
    const std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() < 16 || std::memcmp(data.data(), "CHESSNET", 8) != 0)
        throw std::runtime_error("EncoderNet: " + path_ + " is not a .cnet file");
    std::uint32_t version = 0, count = 0;
    std::memcpy(&version, data.data() + 8, 4);
    std::memcpy(&count, data.data() + 12, 4);
    if (version != 1) throw std::runtime_error("EncoderNet: " + path_ + ": unsupported version");
    if (16 + 56ull * count > data.size()) throw std::runtime_error("EncoderNet: " + path_ + " is truncated");

    std::map<std::string, Entry> entries;
    for (std::uint32_t i = 0; i < count; ++i)
    {
        const char* e = data.data() + 16 + 56 * i;
        Entry entry{};
        std::memcpy(&entry.dtype, e + 32, 4);
        std::memcpy(&entry.offset, e + 40, 8);
        std::memcpy(&entry.bytes, e + 48, 8);
        if (entry.dtype > kI32 || entry.offset % 256 != 0 || entry.offset + entry.bytes > data.size())
            throw std::runtime_error("EncoderNet: " + path_ + ": bad entry table");
        entries[std::string(e, strnlen(e, 32))] = entry;
    }
    const auto get = [&](const std::string& name, std::uint32_t dtype, std::uint64_t n) -> const Entry& {
        const auto it = entries.find(name);
        if (it == entries.end()) throw std::runtime_error("EncoderNet: " + path_ + " has no " + name);
        if (it->second.dtype != dtype || it->second.bytes != n * kDtypeBytes[dtype])
            throw std::runtime_error("EncoderNet: " + path_ + ": " + name + " has the wrong type or size");
        return it->second;
    };
    const auto host_f32 = [&](const std::string& name) {
        float v = 0.f;
        std::memcpy(&v, data.data() + get(name, kF32, 1).offset, 4);
        return v;
    };

    std::int32_t cfg[7];
    std::memcpy(cfg, data.data() + get("config", kI32, 7).offset, sizeof(cfg));
    const int d = cfg[0], layers = cfg[1], heads = cfg[2], d_ff = cfg[3], planes = cfg[4];
    if (d != 128 || heads != 8 || d_ff != 256 || planes != kPackPlanes || layers < 1 || layers > kMaxLayers
        || cfg[5] < 1 || cfg[5] > 2 || cfg[6] < 0 || cfg[6] > 1)
    {
        throw std::runtime_error("EncoderNet: the fused kernel runs d_model 128, 8 heads, d_ff 256 and up to 16 "
                                 "layers; " + path_ + " has d_model " + std::to_string(d) + ", "
                                 + std::to_string(heads) + " heads, d_ff " + std::to_string(d_ff) + ", "
                                 + std::to_string(layers) + " layers");
    }
    params_.layers = layers;
    params_.n_wdl = cfg[5];
    params_.mlhq_from_q = cfg[6];
    const std::uint64_t n_mlp = params_.n_wdl + 1 + params_.mlhq_from_q;
    const std::uint64_t n_out = 3 * params_.n_wdl + 1 + params_.mlhq_from_q;

    check(cudaMalloc(&blob_, data.size()), "cudaMalloc");
    try
    {
        check(cudaMemcpy(blob_, data.data(), data.size(), cudaMemcpyHostToDevice), "upload");
        const auto dev = [&](const std::string& name, std::uint32_t dtype, std::uint64_t n) {
            return static_cast<const void*>(static_cast<const char*>(blob_) + get(name, dtype, n).offset);
        };
        const auto f32 = [&](const std::string& name, std::uint64_t n) {
            return static_cast<const float*>(dev(name, kF32, n));
        };
        params_.embed_w = dev("embed.w", kF16, 128 * 128);
        params_.embed_b = f32("embed.b", 128);
        params_.pos = f32("pos", 64 * 128);
        params_.pair_bias = dev("pair_bias", kF16, 8 * 64 * 64);
        for (int l = 0; l < layers; ++l)
        {
            const std::string p = "l" + std::to_string(l) + ".";
            LayerParams& L = params_.layer[l];
            L.norm1 = f32(p + "norm1", 128);
            L.qkv = dev(p + "qkv", kF16, 384 * 128);
            L.qnorm = f32(p + "qnorm", 16);
            L.knorm = f32(p + "knorm", 16);
            L.out = dev(p + "out", kF16, 128 * 128);
            L.norm2 = f32(p + "norm2", 128);
            L.up = dev(p + "up", kF16, 512 * 128);
            L.down = dev(p + "down", kF16, 128 * 256);
            L.gain = host_f32(p + "gain");
        }
        params_.norm = f32("norm", 128);
        params_.pol_w = dev("pol.w", kF16, 256 * 128);
        params_.pol_b = f32("pol.b", 256);
        params_.promo = dev("promo", kF16, 4 * 128);
        params_.special = host_f32("special");
        params_.vocab = static_cast<const std::uint16_t*>(dev("vocab", kU16, kPolicySize));
        params_.mlp1_w = dev("mlp1.w", kF16, n_mlp * 128 * 128);
        params_.mlp1_b = f32("mlp1.b", n_mlp * 128);
        params_.mlp2_w = f32("mlp2.w", n_out * 128);
        params_.mlp2_b = f32("mlp2.b", n_out);
    }
    catch (...)
    {
        cudaFree(blob_);
        throw;
    }
}

EncoderNet::~EncoderNet()
{
    if (blob_) cudaFree(blob_);
}

cudaError_t EncoderNet::forward(const mcts::single::SmallPackedPosition* d_packed, const float* d_dense, int batch,
                                void* d_out, cudaStream_t stream) const
{
    return launch_encoder_forward(params_, d_packed, d_dense, batch, d_out, stream);
}

std::vector<unsigned char> EncoderNet::run_dense(const float* planes, int batch) const
{
    return run_host(planes, static_cast<size_t>(batch) * kPackElems * sizeof(float), false, batch);
}

std::vector<unsigned char> EncoderNet::run_packed(const mcts::single::SmallPackedPosition* positions,
                                                  int batch) const
{
    return run_host(positions, static_cast<size_t>(batch) * sizeof(*positions), true, batch);
}

std::vector<unsigned char> EncoderNet::run_host(const void* input, size_t bytes, bool packed, int batch) const
{
    const size_t out_bytes = static_cast<size_t>(batch) * kRecordBytes;
    void* d_in = nullptr;
    void* d_out = nullptr;
    std::vector<unsigned char> out(out_bytes);
    cudaError_t err = cudaMalloc(&d_in, bytes);
    if (err == cudaSuccess) err = cudaMalloc(&d_out, out_bytes);
    if (err == cudaSuccess) err = cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
    {
        err = packed ? forward(static_cast<const mcts::single::SmallPackedPosition*>(d_in), nullptr, batch, d_out,
                               nullptr)
                     : forward(nullptr, static_cast<const float*>(d_in), batch, d_out, nullptr);
    }
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    check(err, "run");
    return out;
}

double benchmark(const EncoderNet& net, int batch, int iterations, int streams)
{
    using mcts::single::SmallPackedPosition;
    SmallPackedPosition start{};  // white pieces on ranks 1-2, black on 7-8, every right
    const std::uint64_t pawns = 0xFF00ull, back[6] = {0, 0x42ull, 0x24ull, 0x81ull, 0x08ull, 0x10ull};
    start.bitboards[0][0] = pawns;
    start.bitboards[0][6] = pawns << 40;
    for (int k = 1; k < 6; ++k)
    {
        start.bitboards[0][k] = back[k];
        start.bitboards[0][6 + k] = back[k] << 56;
    }
    start.slot_valid = 1;
    start.stm = 1;
    start.castling = 0xF;
    std::vector<SmallPackedPosition> host(static_cast<size_t>(batch), start);

    std::vector<cudaStream_t> ss(static_cast<size_t>(streams));
    std::vector<void*> d_in(ss.size()), d_out(ss.size()), h_out(ss.size());
    cudaError_t err = cudaSuccess;
    for (size_t i = 0; i < ss.size() && err == cudaSuccess; ++i)
    {
        err = cudaStreamCreateWithFlags(&ss[i], cudaStreamNonBlocking);
        if (err == cudaSuccess) err = cudaMalloc(&d_in[i], host.size() * sizeof(SmallPackedPosition));
        if (err == cudaSuccess) err = cudaMalloc(&d_out[i], host.size() * kRecordBytes);
        if (err == cudaSuccess) err = cudaHostAlloc(&h_out[i], host.size() * kRecordBytes, cudaHostAllocDefault);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_in[i], host.data(), host.size() * sizeof(SmallPackedPosition), cudaMemcpyHostToDevice);
    }
    const auto run = [&](int iters) {
        for (int it = 0; it < iters && err == cudaSuccess; ++it)
            for (size_t i = 0; i < ss.size() && err == cudaSuccess; ++i)
            {
                err = net.forward(static_cast<const SmallPackedPosition*>(d_in[i]), nullptr, batch, d_out[i], ss[i]);
                if (err == cudaSuccess)
                    err = cudaMemcpyAsync(h_out[i], d_out[i], host.size() * kRecordBytes, cudaMemcpyDeviceToHost,
                                          ss[i]);
            }
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
    };
    run(3);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    run(iterations);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    for (size_t i = 0; i < ss.size(); ++i)
    {
        if (ss[i]) cudaStreamDestroy(ss[i]);
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
        cudaFreeHost(h_out[i]);
    }
    check(err, "benchmark");
    return static_cast<double>(batch) * iterations * streams / (ms / 1000.0);
}

} // namespace chessrl::net
