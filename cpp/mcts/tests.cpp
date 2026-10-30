#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "../chess-library/tests/doctest/doctest.hpp"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

#include "inference.hpp"
#include "mcts/cache.hpp"
#include "mcts/common.hpp"

using chessrl::SmallPolicyValue;
using chessrl::mcts::HistoryStack;
using chessrl::mcts::MctsOptions;
using chessrl::mcts::PiecePlanes;
using chessrl::mcts::SearchInput;
using chessrl::mcts::SearchTree;
using chessrl::mcts::TranspositionTable;
using chessrl::mcts::kHistoryDepth;
using chessrl::mcts::snapshot_piece_planes;
using chessrl::mcts::history_stack_push;
using chessrl::mcts::build_root_history_stack;

// -----------------------------------------------------------------------------
// CLI / history parsing
// -----------------------------------------------------------------------------

TEST_CASE("parse_history_argument splits and trims")
{
    const std::string raw = "e2e4, e7e5 ,g1f3";
    const auto result = chessrl::mcts::parse_history_argument(raw);
    REQUIRE(result.size() == 3);
    CHECK(result[0] == "e2e4");
    CHECK(result[1] == "e7e5");
    CHECK(result[2] == "g1f3");
}

TEST_CASE("parse_history_argument handles empty and degenerate input")
{
    CHECK(chessrl::mcts::parse_history_argument("").empty());
    CHECK(chessrl::mcts::parse_history_argument("  ").empty());

    const auto trailing = chessrl::mcts::parse_history_argument("e2e4,,e7e5,");
    REQUIRE(trailing.size() == 2);
    CHECK(trailing[0] == "e2e4");
    CHECK(trailing[1] == "e7e5");
}

TEST_CASE("apply_history reproduces expected board")
{
    SearchInput input;
    input.fen = std::string(chess::constants::STARTPOS);
    input.history = {"e2e4", "c7c5", "g1f3"};

    chess::Board board = chessrl::mcts::apply_history(input);
    // After 3 half-moves it is black-to-move and we are in fullmove 2.
    CHECK(board.getFen() == "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
}

TEST_CASE("apply_history rejects unparseable UCI strings")
{
    SearchInput input;
    input.fen = std::string(chess::constants::STARTPOS);
    input.history = {"zzzz"}; // not parseable as UCI at all

    CHECK_THROWS(chessrl::mcts::apply_history(input));
}

TEST_CASE("parse_cli populates options")
{
    const char* argv[] = {
        "mcts",
        "--fen",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "--history",
        "e2e4,e7e5",
        "--simulations",
        "32",
        "--cpuct",
        "1.2",
        "--temperature-at-root",
        "0.25",
        "--temperature",
        "0.5",
    };
    const int argc = static_cast<int>(std::size(argv));
    auto options = chessrl::mcts::parse_cli(argc, const_cast<char**>(argv));

    CHECK(options.request.fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    REQUIRE(options.request.history.size() == 2);
    CHECK(options.request.history[0] == "e2e4");
    CHECK(options.request.history[1] == "e7e5");
    CHECK(options.simulations == 32);
    CHECK(options.cpuct == doctest::Approx(1.2F));
    CHECK(options.temperature_at_root == doctest::Approx(0.25F));
    CHECK(options.temperature == doctest::Approx(0.5F));
    CHECK(!options.use_dirichlet);
}

TEST_CASE("MctsOptions defaults are pinned")
{
    MctsOptions opts;
    CHECK(opts.simulations == 800);
    CHECK(opts.cpuct == doctest::Approx(1.5F));
    CHECK(opts.temperature == doctest::Approx(1.0F));
    CHECK(opts.temperature_at_root == doctest::Approx(0.0F));
    CHECK(opts.virtual_loss == doctest::Approx(1.0F));
    CHECK(opts.dirichlet_alpha == doctest::Approx(0.3F));
    CHECK(opts.dirichlet_epsilon == doctest::Approx(0.25F));
    CHECK(!opts.use_dirichlet);
    CHECK(opts.small_engine_path == "model.trt");
}

// -----------------------------------------------------------------------------
// Policy vocabulary
// -----------------------------------------------------------------------------

TEST_CASE("policy vocabularies expose expected sizes")
{
    const auto& small_vocab = chessrl::small_policy_vocabulary();
    CHECK(small_vocab.size() == chessrl::small_policy_size());
    CHECK(std::find(small_vocab.begin(), small_vocab.end(), "e2e4") != small_vocab.end());
}

TEST_CASE("policy vocabulary entries are unique and round-trip indices")
{
    const auto& vocab = chessrl::small_policy_vocabulary();
    REQUIRE(vocab.size() == chessrl::small_policy_size());

    // Spot-check a handful of canonical openings/moves.
    for (const std::string& sample : {std::string("e2e4"), std::string("g1f3"), std::string("a1a2"), std::string("h2h4")})
    {
        const auto it = std::find(vocab.begin(), vocab.end(), sample);
        CHECK(it != vocab.end());
        if (it != vocab.end())
        {
            const size_t idx = static_cast<size_t>(std::distance(vocab.begin(), it));
            CHECK(vocab[idx] == sample);
        }
    }
}

// -----------------------------------------------------------------------------
// SearchTree
// -----------------------------------------------------------------------------

TEST_CASE("SearchTree.add_root produces a single-node tree and an empty PV")
{
    SearchTree tree;
    chess::Board board(std::string{chess::constants::STARTPOS});
    const int root = tree.add_root(board, {});
    REQUIRE(root == 0);
    REQUIRE(tree.nodes.size() == 1);
    CHECK(tree.nodes[0].parent == -1);
    CHECK_FALSE(tree.nodes[0].expanded);
    CHECK_FALSE(tree.nodes[0].terminal);

    const auto pv = chessrl::mcts::principal_variation(tree, root, board);
    CHECK(pv.empty());
}

TEST_CASE("SearchTree.add_child wires parent/child relationships")
{
    SearchTree tree;
    chess::Board root_board(std::string{chess::constants::STARTPOS});
    const int root = tree.add_root(root_board, {});

    chess::Board child_board = root_board;
    const chess::Move e2e4 = chess::uci::uciToMove(child_board, "e2e4");
    child_board.makeMove(e2e4);

    const int child = tree.add_child(root, e2e4, /*prior=*/0.5F, child_board);
    REQUIRE(child == 1);
    REQUIRE(tree.nodes.size() == 2);
    CHECK(tree.nodes[child].parent == root);
    CHECK(tree.nodes[child].prior == doctest::Approx(0.5F));
    CHECK(tree.nodes[child].move_from_parent == e2e4);
}

TEST_CASE("SearchTree.add_child rejects an out-of-range parent")
{
    SearchTree tree;
    chess::Board board(std::string{chess::constants::STARTPOS});
    tree.add_root(board, {});
    CHECK_THROWS_AS(tree.add_child(99, chess::Move::NO_MOVE, 0.0F, board), std::out_of_range);
}

// -----------------------------------------------------------------------------
// TranspositionTable
// -----------------------------------------------------------------------------

namespace
{

SmallPolicyValue make_dummy_value(float marker)
{
    SmallPolicyValue v;
    v.policy_logits.assign(8, marker);
    v.wdl = {marker, 0.0F, 0.0F};
    return v;
}

} // namespace

TEST_CASE("TranspositionTable round-trips a value at a single key")
{
    TranspositionTable tt(1); // 1 MB -> at least 1024 entries
    const uint64_t key = 0xDEADBEEFCAFEBABEULL;
    CHECK_FALSE(tt.get(key).has_value());

    tt.put(key, make_dummy_value(0.42F));
    auto out = tt.get(key);
    REQUIRE(out.has_value());
    REQUIRE(out->policy_logits.size() == 8);
    CHECK(out->policy_logits[0] == doctest::Approx(0.42F));
    CHECK(out->wdl[0] == doctest::Approx(0.42F));

    CHECK(tt.hits() == 1);
    CHECK(tt.misses() == 1);
}

TEST_CASE("TranspositionTable concurrent put/get does not corrupt entries")
{
    TranspositionTable tt(2);

    constexpr int kThreads = 4;
    constexpr int kOpsPerThread = 5'000;
    std::atomic<int> errors{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t)
    {
        workers.emplace_back([&, t]() {
            for (int i = 0; i < kOpsPerThread; ++i)
            {
                const uint64_t key = (static_cast<uint64_t>(t) << 32) ^ static_cast<uint64_t>(i);
                const float marker = static_cast<float>(t * 1000 + i);
                tt.put(key, make_dummy_value(marker));
                if (auto v = tt.get(key); v.has_value())
                {
                    // Hash collisions can replace our entry; only validate when
                    // the slot still holds the value we just wrote.
                    if (!v->policy_logits.empty() && v->policy_logits[0] == marker)
                    {
                        if (v->wdl[0] != marker)
                        {
                            ++errors;
                        }
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    CHECK(errors.load() == 0);
}

// -----------------------------------------------------------------------------
// MCTS value-handling: terminal_value, wdl_from_terminal,
// detail::backpropagate, detail::refine_wdl_node
// -----------------------------------------------------------------------------

namespace mcts = chessrl::mcts;

TEST_CASE("wdl_from_terminal: side-to-move perspective")
{
    const auto win  = mcts::wdl_from_terminal(+1.0F);
    const auto loss = mcts::wdl_from_terminal(-1.0F);
    const auto draw = mcts::wdl_from_terminal(0.0F);
    CHECK(win[0]  == doctest::Approx(1.0F)); CHECK(win[1]  == doctest::Approx(0.0F)); CHECK(win[2]  == doctest::Approx(0.0F));
    CHECK(draw[0] == doctest::Approx(0.0F)); CHECK(draw[1] == doctest::Approx(1.0F)); CHECK(draw[2] == doctest::Approx(0.0F));
    CHECK(loss[0] == doctest::Approx(0.0F)); CHECK(loss[1] == doctest::Approx(0.0F)); CHECK(loss[2] == doctest::Approx(1.0F));
}

namespace
{
// Apply a sequence of UCI moves to a board.
chess::Board play_moves(const std::vector<std::string>& uci)
{
    chess::Board b{std::string{chess::constants::STARTPOS}};
    for (const auto& m : uci) { b.makeMove(chess::uci::uciToMove(b, m)); }
    return b;
}
} // namespace

TEST_CASE("terminal_value: Fool's mate is LOSE for side to move (white)")
{
    // 1.f3 e5 2.g4 Qh4#  — after black's third move, white is to move and checkmated.
    chess::Board b = play_moves({"f2f3", "e7e5", "g2g4", "d8h4"});
    REQUIRE(b.isGameOver().second == chess::GameResult::LOSE);
    auto v = mcts::terminal_value(b);
    REQUIRE(v.has_value());
    CHECK(*v == doctest::Approx(-1.0F));
}

TEST_CASE("terminal_value: stalemate is DRAW (0)")
{
    // K vs K+Q stalemate: black to move, no legal move, not in check.
    // Black king a8, white king c6, white queen c7.
    chess::Board b{std::string{"k7/2Q5/2K5/8/8/8/8/8 b - - 0 1"}};
    REQUIRE(b.isGameOver().second == chess::GameResult::DRAW);
    auto v = mcts::terminal_value(b);
    REQUIRE(v.has_value());
    CHECK(*v == doctest::Approx(0.0F));
}

TEST_CASE("terminal_value: insufficient material (K vs K) is DRAW")
{
    chess::Board b{std::string{"k7/8/8/8/8/8/8/K7 w - - 0 1"}};
    REQUIRE(b.isGameOver().second == chess::GameResult::DRAW);
    CHECK(*mcts::terminal_value(b) == doctest::Approx(0.0F));
}

TEST_CASE("terminal_value: startpos is non-terminal (nullopt)")
{
    chess::Board b{std::string{chess::constants::STARTPOS}};
    CHECK(!mcts::terminal_value(b).has_value());
}

TEST_CASE("detail::backpropagate flips the sign at every ply")
{
    // Build a 3-deep linear path: root -> child -> grandchild.
    mcts::SearchTree tree;
    chess::Board root_board{std::string{chess::constants::STARTPOS}};
    const int root = tree.add_root(root_board, {});

    chess::Board after_e2e4 = root_board;
    chess::Move e2e4 = chess::uci::uciToMove(after_e2e4, "e2e4");
    after_e2e4.makeMove(e2e4);
    const int child = tree.add_child(root, e2e4, /*prior=*/0.5F, after_e2e4);

    chess::Board after_e7e5 = after_e2e4;
    chess::Move e7e5 = chess::uci::uciToMove(after_e7e5, "e7e5");
    after_e7e5.makeMove(e7e5);
    const int grand = tree.add_child(child, e7e5, /*prior=*/0.5F, after_e7e5);

    mcts::detail::backpropagate(tree, {root, child, grand}, +0.4F);

    // path is iterated in reverse, so the leaf gets +value, child gets -value,
    // root gets +value.
    CHECK(tree.nodes[grand].value_sum == doctest::Approx(+0.4F));
    CHECK(tree.nodes[child].value_sum == doctest::Approx(-0.4F));
    CHECK(tree.nodes[root].value_sum  == doctest::Approx(+0.4F));
    CHECK(tree.nodes[root].visit_count == 1);
    CHECK(tree.nodes[child].visit_count == 1);
    CHECK(tree.nodes[grand].visit_count == 1);
}

TEST_CASE("detail::refine_wdl_node flips child WDL into parent and visit-weights")
{
    // Synthetic two-child tree.
    //   root (no terminal)
    //    ├── child A: visits=3, terminal LOSS for A → wdl={0,0,1}; flips into root as a WIN.
    //    └── child B: visits=1, terminal DRAW → wdl={0,1,0}; flips into root as DRAW.
    // Expected root.wdl = (3*[1,0,0] + 1*[0,1,0]) / 4 = [0.75, 0.25, 0.0]
    mcts::SearchTree tree;
    chess::Board startpos{std::string{chess::constants::STARTPOS}};
    const int root = tree.add_root(startpos, {});

    // Add a placeholder child by playing e2e4.
    chess::Board ba = startpos;
    chess::Move e2e4 = chess::uci::uciToMove(ba, "e2e4");
    ba.makeMove(e2e4);
    const int a = tree.add_child(root, e2e4, 0.5F, ba);
    chess::Board bb = startpos;
    chess::Move d2d4 = chess::uci::uciToMove(bb, "d2d4");
    bb.makeMove(d2d4);
    const int b = tree.add_child(root, d2d4, 0.5F, bb);

    // Manually set children to look like terminal leaves for the test.
    tree.nodes[a].terminal = true;
    tree.nodes[a].wdl = {0.0F, 0.0F, 1.0F};  // LOSE for child A
    tree.nodes[a].visit_count = 3;
    tree.nodes[b].terminal = true;
    tree.nodes[b].wdl = {0.0F, 1.0F, 0.0F};  // DRAW
    tree.nodes[b].visit_count = 1;

    std::vector<bool> visited(tree.nodes.size(), false);
    auto refined = mcts::detail::refine_wdl_node(tree, root, visited);

    CHECK(refined[0] == doctest::Approx(0.75F));
    CHECK(refined[1] == doctest::Approx(0.25F));
    CHECK(refined[2] == doctest::Approx(0.0F));
    CHECK(tree.nodes[root].wdl[0] == doctest::Approx(0.75F));
}

TEST_CASE("detail::refine_wdl_node leaves unvisited subtrees alone")
{
    mcts::SearchTree tree;
    chess::Board startpos{std::string{chess::constants::STARTPOS}};
    const int root = tree.add_root(startpos, {});
    tree.nodes[root].wdl = {0.4F, 0.3F, 0.3F};  // NN prior

    // Add a child but keep visit_count = 0 → root should NOT update from it.
    chess::Board ba = startpos;
    chess::Move e2e4 = chess::uci::uciToMove(ba, "e2e4");
    ba.makeMove(e2e4);
    const int a = tree.add_child(root, e2e4, 0.5F, ba);
    tree.nodes[a].wdl = {0.9F, 0.05F, 0.05F};
    tree.nodes[a].visit_count = 0;

    std::vector<bool> visited(tree.nodes.size(), false);
    auto refined = mcts::detail::refine_wdl_node(tree, root, visited);

    CHECK(refined[0] == doctest::Approx(0.4F));
    CHECK(refined[1] == doctest::Approx(0.3F));
    CHECK(refined[2] == doctest::Approx(0.3F));
}

TEST_CASE("extract_subtree re-roots a played move's subtree for the next search")
{
    mcts::SearchTree tree;
    chess::Board startpos{std::string{chess::constants::STARTPOS}};
    const int root = tree.add_root(startpos, {});
    chess::Board after_e4 = startpos;
    const chess::Move e2e4 = chess::uci::uciToMove(after_e4, "e2e4");
    after_e4.makeMove(e2e4);
    const int a = tree.add_child(root, e2e4, 0.6F, after_e4);
    chess::Board after_d4 = startpos;
    const chess::Move d2d4 = chess::uci::uciToMove(after_d4, "d2d4");
    after_d4.makeMove(d2d4);
    tree.add_child(root, d2d4, 0.4F, after_d4);
    chess::Board after_e5 = after_e4;
    const chess::Move e7e5 = chess::uci::uciToMove(after_e5, "e7e5");
    after_e5.makeMove(e7e5);
    const int c = tree.add_child(a, e7e5, 0.7F, after_e5);
    chess::Board after_c5 = after_e4;
    const chess::Move c7c5 = chess::uci::uciToMove(after_c5, "c7c5");
    after_c5.makeMove(c7c5);
    const int d = tree.add_child(a, c7c5, 0.3F, after_c5);

    tree.nodes[a].expanded = true;
    tree.nodes[a].visit_count = 3;
    tree.nodes[a].wdl_sum = {1.5F, 0.9F, 0.6F};
    tree.nodes[a].wdl = {0.9F, 0.05F, 0.05F};  // as refine_wdl_node leaves it
    tree.nodes[a].virtual_loss = 2.0F;
    tree.nodes[c].visit_count = 2;
    tree.nodes[c].wdl_sum = {0.4F, 0.6F, 1.0F};
    tree.nodes[c].value_sum = -0.3F;

    const mcts::SearchTree sub = mcts::extract_subtree(tree, a, after_e4.zobrist());
    REQUIRE(sub.nodes.size() == 3);
    CHECK(sub.root_hash == after_e4.zobrist());
    const auto& r = sub.nodes[0];
    CHECK(r.parent == -1);
    CHECK(r.move_from_parent == chess::Move::NO_MOVE);
    CHECK(r.visit_count == 3);
    CHECK(r.virtual_loss == 0.0F);
    CHECK(r.wdl[0] == doctest::Approx(0.5F));  // wdl_sum / visits, not the refined value
    CHECK(r.history == tree.nodes[a].history);
    REQUIRE(r.children.size() == 2);
    CHECK(r.child_moves[0] == e7e5);
    CHECK(r.child_priors[1] == doctest::Approx(0.3F));
    const auto& sc = sub.nodes[static_cast<size_t>(r.children[0])];
    CHECK(sc.parent == 0);
    CHECK(sc.move_from_parent == e7e5);
    CHECK(sc.visit_count == 2);
    CHECK(sc.value_sum == doctest::Approx(-0.3F));
    CHECK(sc.wdl[2] == doctest::Approx(0.5F));
    CHECK(sub.nodes[static_cast<size_t>(r.children[1])].move_from_parent == c7c5);
    (void)d;

    CHECK(mcts::extract_subtree(tree, 99, 0).nodes.empty());  // out of range: empty tree
}

// -----------------------------------------------------------------------------
// AlphaZero history stack
// -----------------------------------------------------------------------------

TEST_CASE("snapshot_piece_planes mirrors chess::Board bitboards in PNBRQK order")
{
    chess::Board startpos{std::string{chess::constants::STARTPOS}};
    PiecePlanes snap = snapshot_piece_planes(startpos);
    CHECK(snap.valid);
    // White pawns: rank 2 = bits 8..15 = 0xFF00.
    CHECK(snap.bitboards[0] == 0x000000000000FF00ULL);
    // White king: e1 (square 4) = bit 4.
    CHECK(snap.bitboards[5] == (1ULL << 4));
    // Black pawns: rank 7 = bits 48..55 = 0x00FF000000000000.
    CHECK(snap.bitboards[6] == 0x00FF000000000000ULL);
    // Black king: e8 (square 60).
    CHECK(snap.bitboards[11] == (1ULL << 60));
}

TEST_CASE("history_stack_push rotates and drops the oldest slot")
{
    HistoryStack stack{};
    for (int i = 0; i < kHistoryDepth + 3; ++i)
    {
        PiecePlanes snap;
        snap.valid = true;
        snap.bitboards[0] = static_cast<std::uint64_t>(i);
        history_stack_push(stack, snap);
        // Newest is always at slot 0.
        CHECK(stack[0].bitboards[0] == static_cast<std::uint64_t>(i));
    }
    // After kHistoryDepth+3 pushes, slot 0 = last push, slot 7 = push (i = total-8).
    // Total pushes = kHistoryDepth + 3 = 11; slot 0 = 10; slot 7 = 11 - 8 = 3.
    CHECK(stack[0].bitboards[0] == 10ULL);
    CHECK(stack[kHistoryDepth - 1].bitboards[0] == 3ULL);
}

TEST_CASE("build_root_history_stack populates slots 0 and 1 after one move")
{
    chess::Board startpos{std::string{chess::constants::STARTPOS}};
    chess::Board after_e4 = startpos;
    chess::Move e4 = chess::uci::uciToMove(after_e4, "e2e4");
    after_e4.makeMove(e4);

    HistoryStack stack = build_root_history_stack(
        std::string(chess::constants::STARTPOS),
        {"e2e4"},
        after_e4);

    CHECK(stack[0].valid);
    CHECK(stack[1].valid);
    // Slot 0 = position AFTER e4: white pawn on e4 (bit 28).
    CHECK((stack[0].bitboards[0] & (1ULL << 28)) != 0ULL);
    // Slot 0 should NOT have the pawn on e2 anymore.
    CHECK((stack[0].bitboards[0] & (1ULL << 12)) == 0ULL);
    // Slot 1 = startpos: pawn at e2 (bit 12), no pawn at e4.
    CHECK((stack[1].bitboards[0] & (1ULL << 12)) != 0ULL);
    CHECK((stack[1].bitboards[0] & (1ULL << 28)) == 0ULL);
    // Slot 2..7 must be empty.
    for (int s = 2; s < kHistoryDepth; ++s)
    {
        CHECK_FALSE(stack[s].valid);
    }
}

TEST_CASE("build_root_history_stack: empty history yields only slot 0")
{
    chess::Board startpos{std::string{chess::constants::STARTPOS}};
    HistoryStack stack = build_root_history_stack(
        std::string(chess::constants::STARTPOS), {}, startpos);
    CHECK(stack[0].valid);
    for (int s = 1; s < kHistoryDepth; ++s)
    {
        CHECK_FALSE(stack[s].valid);
    }
}

TEST_CASE("build_root_history_stack: knight dance flags current repetition")
{
    // Nf3 Nf6 Ng1 Ng8 = startpos repeats.
    std::vector<std::string> moves{"g1f3", "g8f6", "f3g1", "f6g8"};
    chess::Board board{std::string{chess::constants::STARTPOS}};
    for (const auto& m : moves)
    {
        chess::Move mv = chess::uci::uciToMove(board, m);
        board.makeMove(mv);
    }
    HistoryStack stack = build_root_history_stack(
        std::string(chess::constants::STARTPOS), moves, board);
    CHECK(stack[0].repetition);  // current position is a repeat.
}
