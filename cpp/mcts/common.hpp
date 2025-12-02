#pragma once

#include <array>
#include <string>
#include <vector>
#include <memory>

#include "chess-library/include/chess.hpp"

namespace chessrl::mcts
{

struct SearchInput
{
    std::string fen;
    std::vector<std::string> history;
};

struct HistoryEntry;

using HistoryHandle = std::shared_ptr<const HistoryEntry>;

struct MctsOptions
{
    SearchInput request;
    int simulations{800};
    float cpuct{1.5F};
    float temperature_at_root{0.0F};
    float virtual_loss{1.0F};
    float dirichlet_alpha{0.3F};
    float dirichlet_epsilon{0.25F};
    bool use_dirichlet{false};
    float temperature{1.0F};
    std::string small_engine_path{"model.trt"};
};

MctsOptions parse_cli(int argc, char** argv);

std::vector<std::string> parse_history_argument(const std::string& raw);

chess::Board apply_history(const SearchInput& input);

struct TreeNode
{
    // REMOVED: chess::Board board; - Board state is reconstructed during traversal
    
    int parent{-1};
    chess::Move move_from_parent{chess::Move::NO_MOVE}; // The move that led to this node
    bool terminal{false};
    bool expanded{false};
    float prior{0.0F};
    float value_sum{0.0F};
    float virtual_loss{0.0F};
    int visit_count{0};
    float value_prior{0.0F};
    std::array<float, 3> wdl{{0.0F, 0.0F, 0.0F}};
    HistoryHandle history;
    std::vector<int> children;
    std::vector<chess::Move> child_moves;
    std::vector<float> child_priors;
};

struct SearchTree
{
    std::vector<TreeNode> nodes;

    int add_root(const chess::Board& board, const std::vector<std::string>& history);
    void reserve(size_t count);
    int add_child(
        int parent_index,
        chess::Move move,
        float prior,
        const chess::Board& child_board);
};

std::vector<std::string> principal_variation(const SearchTree& tree, int node_index, const chess::Board& root_board);
std::vector<std::string> history_to_vector(const HistoryHandle& tail);

} // namespace chessrl::mcts


