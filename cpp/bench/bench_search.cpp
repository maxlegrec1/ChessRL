// CPU benchmark of the self-play search: the production search code (mcts_small.cpp) on
// positions with their real game history, with a fake network (fake_network.cpp) in
// place of TensorRT. Reports the CPU time the search spends per network evaluation.
// Built by scripts/build_cpu_bench.sh, run by scripts/bench_cpu_search.sh.
//
//   bench_search --positions data/bench/positions.txt --mode full --threads 8
//
// Prints one JSON line. "digest" hashes every search's root visit counts: an
// optimisation meant to leave the search unchanged must leave it unchanged. --dump
// writes those visit counts per position (the golden file).

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    int sims{-1}; // override the mode's simulations
    int repeat{3};
    int limit{0};
    std::uint64_t seed{1};
    std::string dump;
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
        else if (k == "--sims") a.sims = std::stoi(next());
        else if (k == "--repeat") a.repeat = std::stoi(next());
        else if (k == "--limit") a.limit = std::stoi(next());
        else if (k == "--seed") a.seed = std::stoull(next());
        else if (k == "--dump") a.dump = next();
        else if (k == "--sharpness") a.net.sharpness = std::stof(next());
        else if (k == "--delay-us") a.net.delay_us = std::stoi(next());
        else if (k == "--flush-us") a.net.flush_us = std::stoi(next());
        else if (k == "--max-batch") a.net.max_batch = std::stoi(next());
        else throw std::runtime_error("unknown argument " + k);
    }
    if (a.mode != "full" && a.mode != "fast")
        throw std::runtime_error("--mode must be full or fast");
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

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Args a = parse(argc, argv);
        chessrl::bench::configure_fake_network(a.net);
        const std::vector<Position> positions = load_positions(a.positions, a.limit);
        auto runner = std::make_shared<chessrl::mcts::single::SmallBatchRunner>();

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
            chessrl::mcts::run_small_mcts_batch_shared(std::move(options), runner, nullptr, a.threads);
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
            auto summaries = chessrl::mcts::run_small_mcts_batch_shared(std::move(options), runner, nullptr, a.threads);
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
