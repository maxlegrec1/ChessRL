#pragma once

#include <vector>

#include "mcts/common.hpp"
#include "mcts/summary.hpp"

namespace chessrl::mcts
{

MctsSummary run_small_mcts(MctsOptions options);
std::vector<MctsSummary> run_small_mcts_batch(std::vector<MctsOptions> options);

} // namespace chessrl::mcts


