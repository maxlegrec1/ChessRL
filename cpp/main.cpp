#include <algorithm>
#include <iostream>
#include <string_view>
#include <vector>

#include "mcts/adversarial_runner.hpp"
#include "mcts/leela_runner.hpp"
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
        small_options.request.fen = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4";
        small_options.simulations = 600;
        auto small_summary = chessrl::mcts::run_small_mcts(small_options);
        print_summary("[small]", small_summary);

        chessrl::mcts::MctsOptions leela_options;
        leela_options.request.fen = "";
        leela_options.request.history = {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5","g8f6","e1g1"};
        leela_options.simulations = 32;
        auto leela_summary = chessrl::mcts::run_leela_mcts(leela_options);
        print_summary("[leela]", leela_summary);

        chessrl::mcts::MctsOptions adv_options;
        adv_options.request.fen = "";
        adv_options.request.history = {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5","g8f6","e1g1"};
        adv_options.simulations = 32;
        auto adv_summary = chessrl::mcts::run_adversarial_mcts(adv_options);
        print_summary("[adversarial]", adv_summary);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
