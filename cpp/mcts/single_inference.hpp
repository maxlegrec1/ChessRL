#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "inference.hpp"

namespace chessrl::mcts::single
{

SmallPolicyValue evaluate_small_position(
    std::string_view fen,
    const std::string& engine_path = "model_single.trt");

LeelaPolicyValue evaluate_leela_position(
    std::string_view origin_fen,
    const std::vector<std::string>& move_history,
    const std::string& engine_path = "leela_single.trt");

} // namespace chessrl::mcts::single


