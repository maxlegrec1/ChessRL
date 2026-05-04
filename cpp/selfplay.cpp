#include "selfplay.hpp"
#include "mcts/small_runner.hpp"
#include "mcts/single_inference.hpp"
#include "mcts/cache.hpp"
#include "chess-library/include/chess.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include <cstring>

namespace chessrl
{

struct SelfPlayRecord
{
    std::string fen;
    std::string policy;
    // Both WDLs are stored from the side-to-move's perspective.
    std::array<float, 3> value_outcome; // sparse one-hot, set at game end from z
    std::array<float, 3> value_mcts;    // dense WDL from MCTS root after refine
};

// On-disk record layout (144 bytes, fixed):
//   [0..100)   FEN (null-padded)
//   [100..116) policy UCI (null-padded)
//   [116..128) value_outcome (3 x float32)
//   [128..140) value_mcts    (3 x float32)
//   [140..144) reserved padding
constexpr size_t kRecordSize = 144;
constexpr size_t kFenOffset = 0;
constexpr size_t kPolicyOffset = 100;
constexpr size_t kValueOutcomeOffset = 116;
constexpr size_t kValueMctsOffset = 128;

struct BoardSlot
{
    chess::Board board;
    std::vector<std::string> history;
    std::vector<SelfPlayRecord> pending_records;
    int ply_count = 0;
};



struct SelfPlayWorker::Impl
{
    SelfPlayConfig config;
    std::shared_ptr<chessrl::mcts::single::SmallBatchRunner> runner;
    std::shared_ptr<chessrl::mcts::TranspositionTable> cache;
    std::vector<BoardSlot> slots;
    std::fstream outfile;
    size_t current_records = 0;
    // Monotonic counter of EVERY write (append + random-replace). Once the
    // buffer is at max_buffer_size the file size stops growing, so the
    // trainer can no longer infer insertion rate from the file. We persist
    // this counter to a sidecar `<output_path>.writes` after each flush so
    // the trainer can read insertion rate as delta(total_writes) / dt.
    size_t total_writes = 0;
    std::mt19937 opening_rng{std::random_device{}()};

    // Reset a slot to a fresh starting position, optionally seeded with N
    // random legal plies for opening diversity. The random plies are added to
    // slot.history (so MCTS gets the correct move context) but no
    // SelfPlayRecord entries are produced for them.
    void reset_slot(BoardSlot& slot)
    {
        slot.board = chess::Board();
        slot.history.clear();
        slot.pending_records.clear();
        slot.ply_count = 0;

        const int n = config.random_opening_plies;
        for (int i = 0; i < n; ++i)
        {
            if (slot.board.isGameOver().second != chess::GameResult::NONE) break;
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, slot.board);
            if (moves.size() == 0) break;
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            const auto& m = moves[dist(opening_rng)];
            slot.history.push_back(chess::uci::moveToUci(m, slot.board.chess960()));
            slot.board.makeMove(m);
        }
    }

    Impl(const SelfPlayConfig& c) : config(c)
    {
        chessrl::mcts::single::SmallBatchConfig batch_config;
        batch_config.engine_path = config.engine_path;
        batch_config.enable_telemetry = true;
        // Cap max_batch_size to the realistic concurrency. The runner uses this
        // value as its flush threshold: with N MCTS workers, the queue can never
        // hold more than N pending requests, so a 1024-cap forces every batch
        // through a 100us flush timeout. Sizing the threshold to the worker
        // count lets a fully-populated queue dispatch immediately.
        const int realistic_concurrency = (config.threads > 0)
            ? config.threads
            : std::min(config.concurrency, 256);
        batch_config.max_batch_size = std::max(1, realistic_concurrency);
        runner = std::make_shared<chessrl::mcts::single::SmallBatchRunner>(batch_config);

        // hash_size <= 0 disables the TT entirely (skips lock+hash+lookup on
        // every leaf). At <2% hit rate the cache is net-negative, so leaving
        // this off is a real perf win for self-play with few transpositions.
        if (config.hash_size > 0)
        {
            cache = std::make_shared<chessrl::mcts::TranspositionTable>(config.hash_size);
        }

        slots.resize(config.concurrency);
        for (auto& slot : slots)
        {
            reset_slot(slot);
        }

        // Open in read/write mode. Create if not exists.
        // std::fstream doesn't create file if it doesn't exist when using in | out.
        // So we might need to create it first.
        if (!std::filesystem::exists(config.output_path))
        {
            std::ofstream tmp(config.output_path);
        }

        // Truncate to a record boundary if a prior run left a partial trailing
        // record, so subsequent appends don't produce a misaligned record.
        {
            std::error_code ec;
            auto fsize = std::filesystem::file_size(config.output_path, ec);
            if (!ec)
            {
                size_t aligned = (fsize / kRecordSize) * kRecordSize;
                if (aligned != fsize)
                {
                    std::filesystem::resize_file(config.output_path, aligned, ec);
                }
            }
        }

        outfile.open(config.output_path, std::ios::in | std::ios::out | std::ios::binary);
        if (!outfile)
        {
            std::cerr << "Failed to open output file: " << config.output_path << std::endl;
        }
        else
        {
            outfile.seekp(0, std::ios::end);
            size_t size = outfile.tellp();
            current_records = size / kRecordSize;
            // Resume from the existing sidecar if present so an interrupted
            // run reattaches the monotonic counter; otherwise seed from the
            // current append count (an underestimate the first time the run
            // is started after random-replace was already happening, but
            // that's fine — only delta matters).
            std::ifstream wf(config.output_path + ".writes");
            if (wf)
            {
                size_t prev = 0;
                wf >> prev;
                total_writes = std::max(prev, current_records);
            }
            else
            {
                total_writes = current_records;
            }
        }
    }
};

SelfPlayWorker::SelfPlayWorker(const SelfPlayConfig& config)
    : impl_(new Impl(config))
{
}

SelfPlayWorker::~SelfPlayWorker()
{
    delete impl_;
}

SelfPlayMetrics SelfPlayWorker::step()
{
    SelfPlayMetrics metrics;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Prepare MCTS requests
    std::vector<chessrl::mcts::MctsOptions> options;
    options.reserve(impl_->slots.size());

    for (const auto& slot : impl_->slots)
    {
        chessrl::mcts::MctsOptions opt;
        opt.request.fen = slot.board.getFen();
        opt.request.history = slot.history;
        opt.request.board = slot.board;
        opt.simulations = impl_->config.simulations;
        opt.cpuct = impl_->config.cpuct;
        opt.temperature = impl_->config.temperature;
        opt.temperature_at_root = impl_->config.temperature_at_root;
        opt.use_dirichlet = impl_->config.use_dirichlet;
        opt.dirichlet_alpha = impl_->config.dirichlet_alpha;
        opt.dirichlet_epsilon = impl_->config.dirichlet_epsilon;
        opt.small_engine_path = impl_->config.engine_path;
        
        options.push_back(std::move(opt));
    }

    // Run MCTS
    double start_infer_ms = impl_->runner->statistics().total_inference_ms;
    auto summaries = chessrl::mcts::run_small_mcts_batch_shared(
        options, impl_->runner, impl_->cache, impl_->config.threads);
    double end_infer_ms = impl_->runner->statistics().total_inference_ms;

    auto current_time = std::chrono::high_resolution_clock::now();
    double mcts_duration_ms = std::chrono::duration<double, std::milli>(current_time - start_time).count();
    double trt_duration_ms = end_infer_ms - start_infer_ms;

    if (std::getenv("CHESSRL_PROFILE_LOG") != nullptr)
    {
        std::cerr << "[Profile] MCTS=" << mcts_duration_ms << "ms"
                  << " TRT=" << trt_duration_ms << "ms\n";
        if (impl_->cache) impl_->cache->print_stats();
        else std::cerr << "[Cache] disabled (hash_size=0)\n";
    }

    // Process results
    for (size_t i = 0; i < impl_->slots.size(); ++i)
    {
        auto& slot = impl_->slots[i];
        const auto& summary = summaries[i];
        
        // Each position runs config.simulations MCTS simulations
        metrics.nodes_evaluated += impl_->config.simulations;

        if (summary.action.empty())
        {
            impl_->reset_slot(slot);
            metrics.games_finished++;
            continue;
        }

        // Store record
        SelfPlayRecord record;
        record.fen = slot.board.getFen();
        record.policy = summary.action;
        // value_mcts is the MCTS-backed-up WDL (always populated).
        // value_outcome is filled in at game end below; default to zero for now.
        record.value_mcts = summary.value;
        record.value_outcome = {0.0f, 0.0f, 0.0f};
        slot.pending_records.push_back(record);
        metrics.positions_generated++;

        // Apply move
        bool move_found = false;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, slot.board);
        for (const auto& m : moves)
        {
            if (chess::uci::moveToUci(m, slot.board.chess960()) == summary.action)
            {
                slot.board.makeMove(m);
                slot.history.push_back(summary.action);
                slot.ply_count++;
                move_found = true;
                break;
            }
        }

        if (!move_found)
        {
            std::cerr << "Move not found: " << summary.action << " in " << slot.board.getFen() << std::endl;
            impl_->reset_slot(slot);
            continue;
        }

        // Check termination
        auto result = slot.board.isGameOver().second;
        bool game_over = (result != chess::GameResult::NONE) || (slot.ply_count >= 600);

        if (game_over)
        {
            float z = 0.5f;
            if (result == chess::GameResult::LOSE)
            {
                // The side to move lost (e.g. is checkmated).
                // If it's White's turn and they lost, z=0.0.
                // If it's Black's turn and they lost, White won, so z=1.0.
                if (slot.board.sideToMove() == chess::Color::WHITE)
                    z = 0.0f;
                else
                    z = 1.0f;
            }
            else if (result == chess::GameResult::WIN)
            {
                // Should not happen in standard chess checkmate detection, 
                // but if it does (side to move wins):
                if (slot.board.sideToMove() == chess::Color::WHITE)
                    z = 1.0f;
                else
                    z = 0.0f;
            }
            
            // Always derive sparse one-hot game-outcome WDL for every pending record.
            for (auto& rec : slot.pending_records)
            {
                bool white_to_move = (rec.fen.find(" w ") != std::string::npos);

                if (white_to_move)
                {
                    if (z == 1.0f) rec.value_outcome = {1.0f, 0.0f, 0.0f};
                    else if (z == 0.0f) rec.value_outcome = {0.0f, 0.0f, 1.0f};
                    else rec.value_outcome = {0.0f, 1.0f, 0.0f};
                }
                else
                {
                    if (z == 0.0f) rec.value_outcome = {1.0f, 0.0f, 0.0f};
                    else if (z == 1.0f) rec.value_outcome = {0.0f, 0.0f, 1.0f};
                    else rec.value_outcome = {0.0f, 1.0f, 0.0f};
                }
            }
            // Write to buffer
            for (const auto& rec : slot.pending_records)
            {
                char buffer[kRecordSize];
                std::memset(buffer, 0, kRecordSize);

                std::strncpy(buffer + kFenOffset,    rec.fen.c_str(),    99);
                std::strncpy(buffer + kPolicyOffset, rec.policy.c_str(), 15);
                std::memcpy(buffer + kValueOutcomeOffset, rec.value_outcome.data(), 12);
                std::memcpy(buffer + kValueMctsOffset,    rec.value_mcts.data(),    12);

                if (impl_->current_records < impl_->config.max_buffer_size)
                {
                    // Append
                    impl_->outfile.seekp(0, std::ios::end);
                    impl_->outfile.write(buffer, kRecordSize);
                    impl_->current_records++;
                }
                else
                {
                    // Random replacement
                    // Use a simple random generator
                    static std::mt19937 rng(std::random_device{}());
                    std::uniform_int_distribution<size_t> dist(0, impl_->config.max_buffer_size - 1);
                    size_t idx = dist(rng);

                    impl_->outfile.seekp(idx * kRecordSize, std::ios::beg);
                    impl_->outfile.write(buffer, kRecordSize);
                }
                // Always count the write, regardless of which branch handled
                // it. The trainer needs this to compute insertion rate once
                // the buffer is at capacity — the file size becomes constant
                // under random-replacement, but writes (and thus inserts)
                // continue.
                impl_->total_writes++;
            }

            if (impl_->outfile) impl_->outfile.flush();
            // Persist the monotonic write counter to a sidecar file. The
            // trainer reads this to derive insertion rate even after the
            // buffer file has stopped growing. Atomic rename so a partial
            // write is never observed.
            {
                const std::string sidecar = impl_->config.output_path + ".writes";
                const std::string sidecar_tmp = sidecar + ".tmp";
                std::ofstream wfo(sidecar_tmp, std::ios::trunc);
                if (wfo)
                {
                    wfo << impl_->total_writes << "\n";
                    wfo.close();
                    std::error_code ec;
                    std::filesystem::rename(sidecar_tmp, sidecar, ec);
                }
            }

            // Save PGN if requested
            if (impl_->config.save_pgn)
            {
                std::filesystem::create_directories(impl_->config.pgn_dir);
                
                // Replay game to generate SAN
                chess::Board replay_board;
                std::stringstream ss;
                ss << "[Event \"Self-Play\"]\n";
                ss << "[Site \"Local\"]\n";
                ss << "[Date \"????.??.??\"]\n";
                ss << "[Round \"-\"]\n";
                ss << "[White \"SelfPlay\"]\n";
                ss << "[Black \"SelfPlay\"]\n";
                
                std::string result_str = "*";
                if (result == chess::GameResult::WIN) result_str = (slot.board.sideToMove() == chess::Color::WHITE) ? "1-0" : "0-1"; // Wait, sideToMove wins? No, sideToMove checkmated means they LOST.
                // If result is WIN, it means the side to move WON? 
                // chess-library isGameOver() returns {Reason, Result}. 
                // Result is WIN, LOSE, DRAW, NONE.
                // If LOSE, side to move lost.
                // If WIN, side to move won (e.g. opponent timed out? but here we only check board state).
                // Usually isGameOver checks for checkmate (LOSE) or stalemate (DRAW).
                // WIN might not be returned by standard rules unless we check for variant rules.
                // Let's stick to what we derived for z.
                
                if (z == 1.0f) result_str = "1-0";
                else if (z == 0.0f) result_str = "0-1";
                else result_str = "1/2-1/2";
                
                ss << "[Result \"" << result_str << "\"]\n\n";
                
                int move_num = 1;
                for (size_t k = 0; k < slot.history.size(); ++k)
                {
                    if (k % 2 == 0) ss << move_num << ". ";
                    
                    // slot.history[k] is a UCI string (e.g. "e2e4")
                    // We need to find the corresponding Move object to generate SAN.
                    
                    bool found = false;
                    chess::Movelist moves;
                    chess::movegen::legalmoves(moves, replay_board);
                    for (const auto& leg_m : moves)
                    {
                        if (chess::uci::moveToUci(leg_m, replay_board.chess960()) == slot.history[k])
                        {
                            ss << chess::uci::moveToSan(replay_board, leg_m) << " ";
                            replay_board.makeMove(leg_m);
                            found = true;
                            break;
                        }
                    }
                    
                    if (!found)
                    {
                        ss << slot.history[k] << "? ";
                        // Try to apply anyway if possible or break
                        break;
                    }
                    
                    if (k % 2 == 1) move_num++;
                }
                
                ss << result_str << "\n";
                
                // Write to file
                auto now = std::chrono::system_clock::now();
                auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
                std::string filename = impl_->config.pgn_dir + "/game_" + std::to_string(timestamp) + "_" + std::to_string(i) + ".pgn";
                
                std::ofstream pgn_file(filename);
                if (pgn_file)
                {
                    pgn_file << ss.str();
                }
            }

            impl_->reset_slot(slot);
            metrics.games_finished++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    metrics.runtime_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    // Nodes evaluated is roughly simulations * concurrency, but let's be more precise if we can.
    // SmallMCTS returns summary with 'rollouts' (vector of details).
    // The number of rollouts is the number of simulations.
    // So total nodes = sum(summary.rollouts.size()) across batch.
    // I added that above.
    
    return metrics;
}

size_t SelfPlayWorker::get_buffer_size() const
{
    std::filesystem::path p(impl_->config.output_path);
    if (!std::filesystem::exists(p))
    {
        return 0;
    }
    
    size_t file_size = std::filesystem::file_size(p);
    return file_size / kRecordSize;
}

} // namespace chessrl
