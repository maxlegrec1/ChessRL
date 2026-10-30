#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "chess-library/include/chess.hpp"

namespace chessrl::mcts
{

struct HistoryEntry;

using HistoryHandle = std::shared_ptr<const HistoryEntry>;

// Compact piece-bitboard snapshot of a board position, used as the
// per-history-slot payload for the AlphaZero-style 8-board input stack.
// Layout matches the Python encoder: bitboards[0..5] = white PNBRQK,
// bitboards[6..11] = black PNBRQK. `valid=false` slots encode as zero
// piece planes (the all-zero pattern is the "no-history" signal).
struct PiecePlanes
{
    std::array<std::uint64_t, 12> bitboards{};
    bool valid{false};
    bool repetition{false};
};

// 8-slot stack with slot 0 = current position. Older positions live at
// higher indices (slot 7 = t-7). Used by the C++ encoder to emit the
// 111-plane input for the NN.
inline constexpr int kHistoryDepth = 8;
using HistoryStack = std::array<PiecePlanes, kHistoryDepth>;

struct SearchInput
{
    std::string fen;
    std::vector<std::string> history;
    std::optional<chess::Board> board;
    // Precomputed AlphaZero history stack. When set, the runner uses it
    // directly instead of O(plies) replaying the whole game in
    // build_root_history_stack() on every MCTS call (self-play already
    // maintains the rolling snapshots — see SelfPlayWorker::step). Mirrors
    // the `board` short-circuit in apply_history().
    std::optional<HistoryStack> history_stack;
};

// Snapshot a board's piece bitboards into a PiecePlanes value. Order:
// W:PNBRQK then B:PNBRQK. `repetition` is left untouched — the caller
// sets it from `board.isRepetition()` knowledge of the surrounding
// position chain.
PiecePlanes snapshot_piece_planes(const chess::Board& board);

// Push `snap` onto the front of `stack` (slot 0), shifting older entries
// down by one. The oldest (formerly slot 7) is dropped.
void history_stack_push(HistoryStack& stack, const PiecePlanes& snap);

// Build the initial 8-slot stack for a fresh search: slot 0 is the
// position represented by `root_board`, slots 1..7 are populated by
// replaying the move list backwards through chess::Board.makeMove (the
// HistoryHandle linked list only stores moves, not board states).
HistoryStack build_root_history_stack(
    const std::string& fen,
    const std::vector<std::string>& moves,
    const chess::Board& root_board);

struct SearchTree;

struct MctsOptions
{
    SearchInput request;
    int simulations{800};
    float cpuct{1.5F};
    float temperature_at_root{0.0F};
    // Per-leaf virtual loss weight. Each in-flight visit adds this much
    // "phantom win for the child" to its value_sum, making the parent's
    // PUCT view of that child correspondingly worse — discourages two
    // batched descents from picking the same leaf in the same round.
    // c_vloss = 1.0 is the LeelaChess / KataGo convention.
    float virtual_loss{1.0F};
    // Number of leaves to collect per inference round. width=1 reduces to
    // the single-leaf-per-descent behavior we had before virtual loss was
    // wired up. width>1 lets one MCTS worker contribute multiple leaves
    // to the GPU batch per round — the main throughput lever once the
    // encoder/inference paths are fast.
    int virtual_loss_width{1};
    float dirichlet_alpha{0.3F};
    float dirichlet_epsilon{0.25F};
    bool use_dirichlet{false};
    float temperature{1.0F};
    std::string small_engine_path{"model.trt"};
    // Seed of the search's random numbers (Dirichlet noise, root move sampling).
    // 0 = a fresh random seed per search.
    std::uint64_t seed{0};
    // Tree reuse. reuse_tree: the subtree a previous search kept for the move that was
    // played (MctsSummary::tree), whose root is this search's position. The search continues
    // it: `simulations` is then the number of visits the root's moves end up with, so only the
    // missing ones are run. Ignored unless its root_hash is this position's, and when
    // `simulations` is 0. keep_tree: return the subtree under the move this search plays.
    std::shared_ptr<SearchTree> reuse_tree;
    bool keep_tree{false};
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
    // plies_sum: sum (over visits) of (leaf_plies + depth_from_leaf), where
    // leaf_plies is the NN's raw plies-remaining estimate at the simulation's
    // leaf and depth_from_leaf is the number of plies between this node and
    // that leaf. Divide by visit_count for the visit-weighted average plies-
    // remaining at this node. No sign flip — moves-left is symmetric.
    float plies_sum{0.0F};
    // leaf_plies: this node's raw plies-remaining estimate set on expansion.
    // Computed from the NN's mlh_q head as `mlh_q * 25` (the head outputs
    // plies/25). Terminal nodes use 0.
    float leaf_plies{0.0F};
    HistoryHandle history;
    std::vector<int> children;
    std::vector<chess::Move> child_moves;
    std::vector<float> child_priors;
};

struct SearchTree
{
    std::vector<TreeNode> nodes;
    // Zobrist hash of the root position, set by extract_subtree (0 for a fresh tree).
    std::uint64_t root_hash{0};

    int add_root(const chess::Board& board, const std::vector<std::string>& history);
    void reserve(size_t count);
    int add_child(
        int parent_index,
        chess::Move move,
        float prior,
        const chess::Board& child_board);
};

// The subtree under `new_root` as a tree of its own (new_root becomes node 0), so the next
// search can continue it once its move is played. Visits and values are kept; each node's wdl
// is reset to the running average of its backups (refine_wdl_node overwrites it after a
// search) and virtual losses to 0. `root_hash` is the zobrist hash of the new root position.
SearchTree extract_subtree(const SearchTree& tree, int new_root, std::uint64_t root_hash);

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
// ply) and the leaf's raw plies-remaining estimate. Each ancestor accumulates
// `leaf_plies + depth_from_leaf` so plies_sum / visit_count gives the visit-
// weighted plies-remaining estimate at that node. Maintains node.wdl as the
// running average over visits so that in-tree selection has a meaningful
// per-node WDL.
void backpropagate_full(
    SearchTree& tree,
    const std::vector<int>& path,
    float value,
    std::array<float, 3> leaf_wdl,
    float leaf_plies);

// Post-order WDL aggregation starting at `node_index`. Children's WDL is
// flipped to the parent's perspective and visit-weighted-averaged. Terminal
// and unvisited nodes keep their existing WDL. Exposed for unit tests.
std::array<float, 3> refine_wdl_node(SearchTree& tree, int node_index, std::vector<bool>& visited);

} // namespace detail

} // namespace chessrl::mcts


