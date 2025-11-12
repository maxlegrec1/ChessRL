#pragma once

#include <array>
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

struct LeelaInferenceResult
{
    std::vector<std::string> moves;
    std::vector<std::array<float, 3>> value_winner;
    std::vector<std::array<float, 3>> value_q;
    InferenceTimings timings;
};

SmallInferenceResult small_generate_moves(
    const std::vector<std::string>& fens,
    float temperature,
    int iterations = 1,
    const std::string& engine_path = "model.trt");

LeelaInferenceResult leela_generate_from_move_lists(
    const std::vector<std::string>& origin_fens,
    const std::vector<std::vector<std::string>>& move_lists,
    float temperature,
    int iterations = 1,
    const std::string& engine_path = "leela.trt");

} // namespace chessrl


