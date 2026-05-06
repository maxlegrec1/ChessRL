#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace chessrl
{

struct InferenceTimings
{
    double encodeMs{0.0};
    double inferMs{0.0};
    double sampleMs{0.0};
    double totalMs{0.0};
};

struct SmallInferenceResult
{
    std::vector<std::string> moves;
    std::vector<std::array<float, 3>> wdl;
    InferenceTimings timings;
};

struct SmallPolicyValue
{
    std::vector<float> policy_logits;
    std::array<float, 3> wdl;
    float mlh{0.0F}; // moves-left head: tanh(plies_remaining / 25), in [0, 1]
};

SmallInferenceResult small_generate_moves(
    const std::vector<std::string>& fens,
    float temperature,
    int iterations = 1,
    const std::string& engine_path = "model.trt");

size_t small_policy_size();

const std::vector<std::string>& small_policy_vocabulary();

} // namespace chessrl


