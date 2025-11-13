#pragma once

#include <array>
#include <string>
#include <vector>

#include "chess-library/include/chess.hpp"

namespace chessrl::mcts
{

struct SearchInput
{
    std::string fen;
    std::vector<std::string> history;
};

struct MctsOptions
{
    SearchInput request;
    int simulations{800};
    float cpuct{1.5F};
    float virtual_loss{1.0F};
    float dirichlet_alpha{0.3F};
    float dirichlet_epsilon{0.25F};
    bool use_dirichlet{false};
    float temperature{1.0F};
    std::string small_engine_path{"model_single.trt"};
    std::string leela_engine_path{"leela_single.trt"};
};

MctsOptions parse_cli(int argc, char** argv);

std::vector<std::string> parse_history_argument(const std::string& raw);

chess::Board apply_history(const SearchInput& input);

struct TreeNode
{
    int parent{-1};
    chess::Move move_from_parent{chess::Move::NO_MOVE};
    chess::Board board{};
    bool terminal{false};
    bool expanded{false};
    float prior{0.0F};
    float value_sum{0.0F};
    float virtual_loss{0.0F};
    int visit_count{0};
    float value_prior{0.0F};
    std::array<float, 3> wdl{{0.0F, 0.0F, 0.0F}};
    std::vector<std::string> history;
    std::vector<int> children;
    std::vector<chess::Move> child_moves;
    std::vector<float> child_priors;
};

struct SearchTree
{
    std::vector<TreeNode> nodes;

    int add_root(const chess::Board& board, const std::vector<std::string>& history);
    int add_child(
        int parent_index,
        chess::Move move,
        float prior,
        const chess::Board& child_board,
        const std::string& move_uci);
};

std::array<float, 3> orient_small_wdl(const chess::Board& board, const std::array<float, 3>& raw_wdl);
std::array<float, 3> orient_leela_wdl(const chess::Board& board, const std::array<float, 3>& white_perspective_wdl);
std::vector<std::string> principal_variation(const SearchTree& tree, int node_index);

} // namespace chessrl::mcts


