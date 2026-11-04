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
    std::shared_ptr<TranspositionTable> cache = {},
    int num_threads = 0,  // 0 = auto: min(jobs, 256)
    int searches_per_thread = 1);  // searches each thread interleaves
// One search on a shared runner, in the calling thread (asynchronous self-play).
MctsSummary run_small_mcts_shared(
    MctsOptions options,
    std::shared_ptr<single::SmallBatchRunner> runner,
    std::shared_ptr<TranspositionTable> cache = {});
// With CHESSRL_MCTS_PHASES set: print the search threads' phase times since the last report
// ([mcts-phases], summed over `threads` threads) and reset them.
void report_mcts_phases(int threads);

} // namespace chessrl::mcts


