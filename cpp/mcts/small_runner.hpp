#pragma once

#include <vector>

#include "mcts/cache.hpp"
#include "mcts/common.hpp"
#include "mcts/summary.hpp"
#include "mcts/single_inference.hpp"

namespace chessrl::mcts
{

MctsSummary run_small_mcts(MctsOptions options);
std::vector<MctsSummary> run_small_mcts_batch(std::vector<MctsOptions> options);
std::vector<MctsSummary> run_small_mcts_batch_shared(
    std::vector<MctsOptions> options, 
    std::shared_ptr<single::SmallBatchRunner> runner,
    std::shared_ptr<TranspositionTable> cache = {});

} // namespace chessrl::mcts


