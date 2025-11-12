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

struct MoveGenerationResult
{
    std::vector<std::string> moves;
    std::vector<std::array<float, 3>> wdl;
    InferenceTimings timings;
};

MoveGenerationResult generate_moves(
    const std::vector<std::string>& fens,
    float temperature,
    int iterations = 1,
    const std::string& engine_path = "model.trt");

} // namespace chessrl


