#pragma once

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace chessrl::mcts
{

struct SearchTree;

struct RolloutDetail
{
    std::string root_move;
    int visit_count{0};
    float visit_fraction{0.0F};
    float prior{0.0F};
    float average_value{0.0F};
    std::vector<std::string> continuation;
};

struct MctsSummary
{
    std::string action;
    std::vector<std::pair<std::string, float>> policy;
    std::array<float, 3> value{{0.0F, 0.0F, 0.0F}};
    // Visit-weighted backed-up plies-remaining at the root, in raw plies. Set
    // to a negative sentinel when no NN visits accumulated (e.g. terminal
    // root) so callers can write a "no target" sentinel into the buffer.
    float backed_plies{-1.0F};
    std::vector<RolloutDetail> rollouts;
    // Simulations this search ran, and visits it inherited from a reused tree.
    int new_simulations{0};
    int reused_visits{0};
    // With MctsOptions::keep_tree: the subtree under the move played, for the next search.
    std::shared_ptr<SearchTree> tree;
};

} // namespace chessrl::mcts


