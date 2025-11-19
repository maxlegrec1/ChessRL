#include <algorithm>
#include <iostream>
#include <string_view>
#include <vector>

#include "mcts/small_runner.hpp"
#include "mcts/summary.hpp"

namespace
{

void print_summary(std::string_view label, const chessrl::mcts::MctsSummary& summary)
{
    std::cout << label << " move: " << (summary.action.empty() ? "<none>" : summary.action) << '\n';

    if (!summary.policy.empty())
    {
        std::cout << label << " policy:" << '\n';
        for (const auto& [move, prob] : summary.policy)
        {
            std::cout << "  " << move << ": " << prob << '\n';
        }
    }

    if (!summary.rollouts.empty())
    {
        std::cout << label << " visits:" << '\n';
        for (const auto& detail : summary.rollouts)
        {
            std::cout << "  " << detail.root_move << ": " << detail.visit_count << '\n';
        }
    }
}

} // namespace

int main()
{
    try
    {
        chessrl::mcts::MctsOptions small_options;
        small_options.request.fen = "r1bqkb1r/p4ppp/2p2n2/nB2p1N1/8/5Q2/PPPP1PPP/RNB1K2R b KQkq - 1 8";
        small_options.simulations = 600;
        small_options.cpuct = 1.5F;
        auto small_summary = chessrl::mcts::run_small_mcts(small_options);
        print_summary("[small]", small_summary);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
