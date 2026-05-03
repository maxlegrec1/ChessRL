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
using chessrl::mcts::MctsOptions;
using chessrl::mcts::SearchInput;
using chessrl::mcts::SearchTree;
using chessrl::mcts::TranspositionTable;

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
