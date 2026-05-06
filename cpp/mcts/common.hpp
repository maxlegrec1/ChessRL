#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "chess-library/include/chess.hpp"

namespace chessrl::mcts
{

struct SearchInput
{
    std::string fen;
    std::vector<std::string> history;
    std::optional<chess::Board> board;
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
    int parent{-1};
    chess::Move move_from_parent{chess::Move::NO_MOVE}; // The move that led to this node
    bool terminal{false};
    bool expanded{false};
    float prior{0.0F};
    float value_sum{0.0F};
    float virtual_loss{0.0F};
    int visit_count{0};
    float value_prior{0.0F};
    // wdl: backed-up running average over visits (in this node's side-to-move
    // perspective, with W↔L flips applied on each ply during backprop).
    // Initialized at expansion to the NN's leaf prediction so the very first
    // visit already has a meaningful triple.
    std::array<float, 3> wdl{{0.0F, 0.0F, 0.0F}};
    // wdl_sum: sum (over visits) of leaf-WDL triples flipped to this node's
    // perspective. Divide by visit_count for the running average.
    std::array<float, 3> wdl_sum{{0.0F, 0.0F, 0.0F}};
    // mlh_sum: sum (over visits) of leaf MLH values (tanh-saturated, in [0,1]).
    // No sign flip — moves-left is symmetric across sides. Divide by
    // visit_count for the running average.
    float mlh_sum{0.0F};
    // mlh value at this leaf (set on expansion from the NN). For terminal
    // nodes this is 0 (game is over from here, zero plies remaining).
    float leaf_mlh{0.0F};
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

// Returns the terminal value of `board` from the side-to-move's perspective:
// +1 = win, -1 = loss (checkmate), 0 = draw, nullopt = non-terminal. Note that
// chess::Board::isGameOver() never returns GameResult::WIN through standard
// rules (a side-to-move that is checkmated has lost), so the +1 path is
// defensive only.
std::optional<float> terminal_value(const chess::Board& board);

// Map a scalar terminal value (in side-to-move perspective: +1 win, -1 loss,
// 0 draw) to a one-hot WDL triple.
std::array<float, 3> wdl_from_terminal(float v);

namespace detail
{

// Backpropagate a side-to-move-relative scalar value up `path`, flipping the
// sign at every ply. Exposed for unit tests; production code calls the same
// logic from inside the MCTS runner.
void backpropagate(SearchTree& tree, const std::vector<int>& path, float value);

// Extended backprop: also propagate the leaf's WDL triple (with W↔L flip per
// ply) and MLH scalar (no flip). Maintains node.wdl as the running average
// over visits so that in-tree selection has a meaningful per-node WDL.
void backpropagate_full(
    SearchTree& tree,
    const std::vector<int>& path,
    float value,
    std::array<float, 3> leaf_wdl,
    float leaf_mlh);

// Post-order WDL aggregation starting at `node_index`. Children's WDL is
// flipped to the parent's perspective and visit-weighted-averaged. Terminal
// and unvisited nodes keep their existing WDL. Exposed for unit tests.
std::array<float, 3> refine_wdl_node(SearchTree& tree, int node_index, std::vector<bool>& visited);

} // namespace detail

} // namespace chessrl::mcts


