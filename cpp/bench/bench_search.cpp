// CPU benchmark of the self-play search: the production search code (mcts_small.cpp) on
// positions with their real game history, with a fake network (fake_network.cpp) in
// place of the CUDA one. Reports the CPU time the search spends per network evaluation.
// Built by scripts/build_cpu_bench.sh, run by scripts/bench_cpu_search.sh.
//
//   bench_search --positions data/bench/positions.txt --mode full --threads 8
//   bench_search --mode selfplay --boards 256 --steps 150 --tree-reuse 1
//
// Prints one JSON line. "digest" hashes every search's root visit counts: an
// optimisation meant to leave the search unchanged must leave it unchanged. --dump
// writes those visit counts per position (the golden file). The selfplay mode runs the
// real SelfPlayWorker (cpp/selfplay.cpp) with the recipe's settings and reports network
// evaluations per move played, e.g. with and without tree reuse.

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bench/fake_network.hpp"
#include "chess-library/include/chess.hpp"
#include "mcts/common.hpp"
#include "mcts/single_inference.hpp"
#include "mcts/small_runner.hpp"
#include "selfplay.hpp"

namespace
{

using chessrl::mcts::HistoryStack;
using chessrl::mcts::MctsOptions;
using chessrl::mcts::MctsSummary;

struct Args
{
    std::string positions{"data/bench/positions.txt"};
    std::string mode{"full"}; // full (400 sims, noise) or fast (50 sims) as in the recipe
    int threads{8};
    int searches_per_thread{1};
    int sims{-1}; // override the mode's simulations
    int repeat{3};
    int limit{0};
    std::uint64_t seed{1};
    std::string dump;
    // selfplay mode
    int boards{256};
    int steps{150};
    int warmup_steps{30};
    int fast_sims{50};
    bool tree_reuse{false};
    std::string buffer{"cpp/build/bench/selfplay_bench.bin"};
    chessrl::bench::FakeNetworkConfig net;
};

Args parse(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--positions") a.positions = next();
        else if (k == "--mode") a.mode = next();
        else if (k == "--threads") a.threads = std::stoi(next());
        else if (k == "--searches-per-thread") a.searches_per_thread = std::stoi(next());
        else if (k == "--sims") a.sims = std::stoi(next());
        else if (k == "--repeat") a.repeat = std::stoi(next());
        else if (k == "--limit") a.limit = std::stoi(next());
        else if (k == "--seed") a.seed = std::stoull(next());
        else if (k == "--dump") a.dump = next();
        else if (k == "--sharpness") a.net.sharpness = std::stof(next());
        else if (k == "--delay-us") a.net.delay_us = std::stoi(next());
        else if (k == "--flush-us") a.net.flush_us = std::stoi(next());
        else if (k == "--max-batch") a.net.max_batch = std::stoi(next());
        else if (k == "--boards") a.boards = std::stoi(next());
        else if (k == "--steps") a.steps = std::stoi(next());
        else if (k == "--warmup-steps") a.warmup_steps = std::stoi(next());
        else if (k == "--fast-sims") a.fast_sims = std::stoi(next());
        else if (k == "--tree-reuse") a.tree_reuse = std::stoi(next()) != 0;
        else if (k == "--buffer") a.buffer = next();
        else throw std::runtime_error("unknown argument " + k);
    }
    if (a.mode != "full" && a.mode != "fast" && a.mode != "selfplay" && a.mode != "reusecheck")
        throw std::runtime_error("--mode must be full, fast, selfplay or reusecheck");
    return a;
}

struct Position
{
    chess::Board board;
    std::vector<std::string> moves; // UCI from the standard start
    HistoryStack history;
};

// One position per line: its game's UCI moves from the standard start ('#' = comment).
std::vector<Position> load_positions(const std::string& path, int limit)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<Position> out;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        Position p;
        std::istringstream words(line);
        std::string uci;
        while (words >> uci)
        {
            const chess::Move move = chess::uci::uciToMove(p.board, uci);
            p.board.makeMove(move);
            p.moves.push_back(uci);
        }
        p.history = chessrl::mcts::build_root_history_stack(
            std::string(chess::constants::STARTPOS), p.moves, p.board);
        out.push_back(std::move(p));
        if (limit > 0 && static_cast<int>(out.size()) >= limit)
            break;
    }
    return out;
}

// The recipe's settings (src/train/config.yaml, selfplay), as SelfPlayWorker::step sets them.
MctsOptions make_options(const Position& p, const Args& a, size_t index)
{
    const bool full = a.mode == "full";
    MctsOptions opt;
    opt.request.fen = p.board.getFen();
    opt.request.history = p.moves;
    opt.request.board = p.board;
    opt.request.history_stack = p.history;
    opt.simulations = a.sims >= 0 ? a.sims : (full ? 400 : 50);
    opt.cpuct = 0.65F;
    opt.temperature = 1.3F;
    opt.temperature_at_root = p.moves.size() < 30 ? 1.0F : 0.0F;
    opt.virtual_loss_width = 8;
    opt.use_dirichlet = full;
    opt.dirichlet_alpha = 0.3F;
    opt.dirichlet_epsilon = 0.25F;
    opt.seed = a.seed * 1000003ULL + index + 1;
    return opt;
}

double cpu_seconds()
{
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return static_cast<double>(r.ru_utime.tv_sec + r.ru_stime.tv_sec)
        + static_cast<double>(r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6;
}

std::uint64_t fnv(std::uint64_t h, const std::string& s)
{
    for (unsigned char c : s)
    {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

std::uint64_t digest(const std::vector<MctsSummary>& summaries)
{
    std::uint64_t h = 1469598103934665603ULL;
    for (const auto& s : summaries)
    {
        std::vector<std::pair<std::string, int>> visits;
        for (const auto& r : s.rollouts)
            visits.emplace_back(r.root_move, r.visit_count);
        std::sort(visits.begin(), visits.end());
        for (const auto& [move, n] : visits)
            h = fnv(h, move + ":" + std::to_string(n) + ";");
        h = fnv(h, "|" + s.action + "|");
    }
    return h;
}

// exp(entropy) of the root priors: the number of moves the network effectively considers.
double effective_moves(const MctsSummary& s)
{
    double total = 0.0;
    for (const auto& r : s.rollouts)
        total += r.prior;
    if (total <= 0.0)
        return 0.0;
    double entropy = 0.0;
    for (const auto& r : s.rollouts)
    {
        const double p = r.prior / total;
        if (p > 0.0)
            entropy -= p * std::log(p);
    }
    return std::exp(entropy);
}

void remove_buffer(const std::string& path)
{
    for (const char* suffix : {"", ".writes", ".stats.json"})
        std::filesystem::remove(path + suffix);
}

// The real SelfPlayWorker with the recipe's settings (src/train/config.yaml, selfplay):
// network evaluations per move played over `steps` worker steps, after `warmup_steps`.
int run_selfplay(const Args& a)
{
    chessrl::SelfPlayConfig cfg;
    cfg.engine_path = "fake";
    cfg.output_path = a.buffer;
    cfg.concurrency = a.boards;
    cfg.threads = a.threads;
    cfg.hash_size = 0;
    cfg.simulations = a.sims >= 0 ? a.sims : 400;
    cfg.pcr_full_prob = 0.25F;
    cfg.pcr_fast_simulations = a.fast_sims;
    cfg.cpuct = 0.65F;
    cfg.temperature = 1.3F;
    cfg.temperature_at_root = 1.0F;
    cfg.temperature_plies = 30;
    cfg.temperature_late = 0.0F;
    cfg.virtual_loss_width = 8;
    cfg.use_dirichlet = true;
    cfg.dirichlet_alpha = 0.3F;
    cfg.dirichlet_epsilon = 0.25F;
    cfg.random_opening_plies = 8;
    cfg.resign_threshold = 0.05F;
    cfg.resign_consecutive = 3;
    cfg.resign_playthrough = 0.1F;
    cfg.max_plies = 600;
    cfg.max_buffer_size = 200000;
    cfg.fifo_buffer = true;
    cfg.tree_reuse = a.tree_reuse;
    std::filesystem::create_directories(std::filesystem::path(a.buffer).parent_path());
    remove_buffer(a.buffer);

    std::uint64_t evals = 0;
    double wall = 0.0;
    double cpu = 0.0;
    chessrl::SelfPlayStats s0;
    chessrl::SelfPlayStats s1;
    {
        chessrl::SelfPlayWorker worker(cfg);
        for (int i = 0; i < a.warmup_steps; ++i)
            worker.step();
        s0 = worker.stats();
        const auto e0 = chessrl::bench::fake_network_stats().evaluations;
        const double c0 = cpu_seconds();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < a.steps; ++i)
            worker.step();
        wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        cpu = cpu_seconds() - c0;
        evals = chessrl::bench::fake_network_stats().evaluations - e0;
        s1 = worker.stats();
    }
    remove_buffer(a.buffer);
    const auto moves = static_cast<std::uint64_t>((s1.full_searches - s0.full_searches)
        + (s1.fast_searches - s0.fast_searches) + (s1.policy_moves - s0.policy_moves));
    char line[1024];
    std::snprintf(line, sizeof line,
        "{\"mode\":\"selfplay\",\"tree_reuse\":%s,\"boards\":%d,\"threads\":%d,\"steps\":%d,"
        "\"moves\":%llu,\"full\":%lld,\"games\":%lld,\"records\":%lld,\"evals\":%llu,"
        "\"evals_per_move\":%.2f,\"simulations_run\":%lld,\"reused_visits\":%lld,"
        "\"wall_s\":%.2f,\"cpu_us_per_eval\":%.2f,\"avg_game_plies\":%.1f,"
        "\"ends\":{\"checkmate\":%lld,\"stalemate\":%lld,\"repetition\":%lld,\"fifty\":%lld,"
        "\"insufficient\":%lld,\"max_plies\":%lld,\"resign\":%lld}}",
        a.tree_reuse ? "true" : "false", a.boards, a.threads, a.steps,
        static_cast<unsigned long long>(moves),
        static_cast<long long>(s1.full_searches - s0.full_searches),
        static_cast<long long>(s1.games - s0.games),
        static_cast<long long>(s1.records_written - s0.records_written),
        static_cast<unsigned long long>(evals),
        moves ? static_cast<double>(evals) / static_cast<double>(moves) : 0.0,
        static_cast<long long>(s1.simulations_run - s0.simulations_run),
        static_cast<long long>(s1.reused_visits - s0.reused_visits), wall,
        evals ? cpu * 1e6 / static_cast<double>(evals) : 0.0,
        s1.games > s0.games ? static_cast<double>(s1.game_plies - s0.game_plies)
                / static_cast<double>(s1.games - s0.games) : 0.0,
        static_cast<long long>(s1.end_checkmate - s0.end_checkmate),
        static_cast<long long>(s1.end_stalemate - s0.end_stalemate),
        static_cast<long long>(s1.end_repetition - s0.end_repetition),
        static_cast<long long>(s1.end_fifty - s0.end_fifty),
        static_cast<long long>(s1.end_insufficient - s0.end_insufficient),
        static_cast<long long>(s1.end_max_plies - s0.end_max_plies),
        static_cast<long long>(s1.end_resign - s0.end_resign));
    std::cout << line << std::endl;
    return 0;
}

// Tree-reuse check (deterministic: no root noise, most visited move, fake network). For each
// position: search it keeping the subtree of the chosen move; from the position after that move,
// compare a fresh 400-simulation search, a reused one (400 visits in all) and a fresh 800 one.
// A correct reuse agrees with fresh 400 about as often as fresh 800 does, and its root's moves
// end up with exactly max(400, inherited) visits.
int run_reusecheck(const std::vector<Position>& positions, const Args& a,
                   const std::shared_ptr<chessrl::mcts::single::SmallBatchRunner>& runner)
{
    auto options_at = [&](const chess::Board& board, const std::vector<std::string>& moves, int sims) {
        MctsOptions opt;
        opt.request.fen = board.getFen();
        opt.request.history = moves;
        opt.request.board = board;
        opt.request.history_stack = chessrl::mcts::build_root_history_stack(
            std::string(chess::constants::STARTPOS), moves, board);
        opt.simulations = sims;
        opt.cpuct = 0.65F;
        opt.temperature = 1.3F;
        opt.temperature_at_root = 0.0F;
        opt.virtual_loss_width = 8;
        opt.use_dirichlet = false;
        opt.seed = 1;
        return opt;
    };
    std::vector<MctsOptions> parents;
    for (const auto& p : positions)
    {
        parents.push_back(options_at(p.board, p.moves, 400));
        parents.back().keep_tree = true;
    }
    auto first = chessrl::mcts::run_small_mcts_batch_shared(std::move(parents), runner, nullptr, a.threads, a.searches_per_thread);

    std::vector<MctsOptions> fresh, reused, fresh800;
    for (size_t i = 0; i < positions.size(); ++i)
    {
        if (!first[i].tree)
            continue;
        chess::Board board = positions[i].board;
        board.makeMove(chess::uci::uciToMove(board, first[i].action));
        std::vector<std::string> moves = positions[i].moves;
        moves.push_back(first[i].action);
        fresh.push_back(options_at(board, moves, 400));
        fresh800.push_back(options_at(board, moves, 800));
        reused.push_back(options_at(board, moves, 400));
        reused.back().reuse_tree = first[i].tree;
    }
    const size_t n = fresh.size();
    auto rf = chessrl::mcts::run_small_mcts_batch_shared(std::move(fresh), runner, nullptr, a.threads, a.searches_per_thread);
    auto rr = chessrl::mcts::run_small_mcts_batch_shared(std::move(reused), runner, nullptr, a.threads, a.searches_per_thread);
    auto r8 = chessrl::mcts::run_small_mcts_batch_shared(std::move(fresh800), runner, nullptr, a.threads, a.searches_per_thread);

    size_t agree_reused = 0, agree_800 = 0, budget_ok = 0;
    double inherited = 0.0, new_sims = 0.0, value_gap_reused = 0.0, value_gap_800 = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        agree_reused += rr[i].action == rf[i].action;
        agree_800 += r8[i].action == rf[i].action;
        int visits = 0;
        for (const auto& r : rr[i].rollouts)
            visits += r.visit_count;
        budget_ok += visits == std::max(400, rr[i].reused_visits);
        inherited += rr[i].reused_visits;
        new_sims += rr[i].new_simulations;
        const auto ev = [](const MctsSummary& s) { return s.value[0] + 0.5 * s.value[1]; };
        value_gap_reused += std::abs(ev(rr[i]) - ev(rf[i]));
        value_gap_800 += std::abs(ev(r8[i]) - ev(rf[i]));
    }
    std::printf("{\"mode\":\"reusecheck\",\"positions\":%zu,\"agree_reused_vs_fresh400\":%.3f,"
                "\"agree_fresh800_vs_fresh400\":%.3f,\"budget_exact\":%.3f,\"mean_inherited\":%.1f,"
                "\"mean_new_sims\":%.1f,\"value_gap_reused\":%.4f,\"value_gap_800\":%.4f}\n",
                n, n ? static_cast<double>(agree_reused) / n : 0.0,
                n ? static_cast<double>(agree_800) / n : 0.0,
                n ? static_cast<double>(budget_ok) / n : 0.0, n ? inherited / n : 0.0,
                n ? new_sims / n : 0.0, n ? value_gap_reused / n : 0.0, n ? value_gap_800 / n : 0.0);
    return budget_ok == n ? 0 : 4;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Args a = parse(argc, argv);
        chessrl::bench::configure_fake_network(a.net);
        if (a.mode == "selfplay")
            return run_selfplay(a);
        const std::vector<Position> positions = load_positions(a.positions, a.limit);
        auto runner = std::make_shared<chessrl::mcts::single::SmallBatchRunner>();
        if (a.mode == "reusecheck")
            return run_reusecheck(positions, a, runner);

        auto build = [&]() {
            std::vector<MctsOptions> options;
            options.reserve(positions.size());
            for (size_t i = 0; i < positions.size(); ++i)
                options.push_back(make_options(positions[i], a, i));
            return options;
        };

        // Warm-up (vocabulary, allocator), not timed.
        {
            auto options = build();
            options.resize(std::min<size_t>(options.size(), 16));
            chessrl::mcts::run_small_mcts_batch_shared(std::move(options), runner, nullptr, a.threads, a.searches_per_thread);
        }

        double wall = 0.0;
        double cpu = 0.0;
        std::uint64_t evals = 0;
        std::uint64_t first_digest = 0;
        bool deterministic = true;
        std::vector<MctsSummary> kept;
        for (int r = 0; r < a.repeat; ++r)
        {
            auto options = build();
            const auto e0 = chessrl::bench::fake_network_stats().evaluations;
            const double c0 = cpu_seconds();
            const auto t0 = std::chrono::steady_clock::now();
            auto summaries = chessrl::mcts::run_small_mcts_batch_shared(std::move(options), runner, nullptr, a.threads, a.searches_per_thread);
            const auto t1 = std::chrono::steady_clock::now();
            cpu += cpu_seconds() - c0;
            wall += std::chrono::duration<double>(t1 - t0).count();
            evals += chessrl::bench::fake_network_stats().evaluations - e0;
            const std::uint64_t d = digest(summaries);
            if (r == 0)
            {
                first_digest = d;
                kept = std::move(summaries);
            }
            else if (d != first_digest)
            {
                deterministic = false;
            }
        }

        std::vector<chessrl::mcts::single::SmallPackedPosition> packed;
        for (const auto& p : positions)
            packed.push_back(chessrl::mcts::single::pack_small_position(p.board, p.history));
        const double fake_ns = chessrl::bench::fake_network_cost_ns(packed, 200);

        double eff = 0.0;
        for (const auto& s : kept)
            eff += effective_moves(s);
        eff /= std::max<size_t>(1, kept.size());

        const double cpu_us = evals ? cpu * 1e6 / static_cast<double>(evals) : 0.0;
        const double fake_us = a.net.delay_us > 0 ? 0.0 : fake_ns / 1000.0;
        char line[1024];
        std::snprintf(line, sizeof line,
            "{\"mode\":\"%s\",\"threads\":%d,\"positions\":%zu,\"sims\":%d,\"repeat\":%d,"
            "\"sharpness\":%.2f,\"delay_us\":%d,\"evals\":%llu,\"wall_s\":%.3f,\"cpu_s\":%.3f,"
            "\"evals_per_s\":%.0f,\"cpu_us_per_eval\":%.3f,\"fake_us_per_eval\":%.3f,"
            "\"search_us_per_eval\":%.3f,\"effective_moves\":%.2f,\"digest\":\"%016llx\","
            "\"deterministic\":%s}",
            a.mode.c_str(), a.threads, positions.size(),
            a.sims >= 0 ? a.sims : (a.mode == "full" ? 400 : 50), a.repeat,
            a.net.sharpness, a.net.delay_us, static_cast<unsigned long long>(evals), wall, cpu,
            wall > 0 ? static_cast<double>(evals) / wall : 0.0, cpu_us, fake_us, cpu_us - fake_us, eff,
            static_cast<unsigned long long>(first_digest), deterministic ? "true" : "false");
        std::cout << line << std::endl;

        if (!a.dump.empty())
        {
            std::ofstream out(a.dump);
            for (size_t i = 0; i < kept.size(); ++i)
            {
                const auto& s = kept[i];
                out << "{\"index\":" << i << ",\"ply\":" << positions[i].moves.size() << ",\"mode\":\""
                    << a.mode << "\",\"action\":\"" << s.action << "\",\"visits\":{";
                std::vector<std::pair<int, std::string>> visits;
                for (const auto& r : s.rollouts)
                    visits.emplace_back(-r.visit_count, r.root_move);
                std::sort(visits.begin(), visits.end());
                for (size_t k = 0; k < visits.size(); ++k)
                    out << (k ? "," : "") << "\"" << visits[k].second << "\":" << -visits[k].first;
                out << "}}\n";
            }
        }
        return deterministic ? 0 : 3;
    }
    catch (const std::exception& e)
    {
        std::cerr << "bench_search: " << e.what() << "\n";
        return 2;
    }
}
