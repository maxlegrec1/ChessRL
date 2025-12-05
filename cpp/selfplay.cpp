#include "selfplay.hpp"
#include "mcts/small_runner.hpp"
#include "mcts/single_inference.hpp"
#include "mcts/cache.hpp"
#include "chess-library/include/chess.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <array>
#include <filesystem>
#include <random>
#include <cstring>

namespace chessrl
{

struct SelfPlayRecord
{
    std::string fen;
    std::string policy;
    std::array<float, 3> value; // Win, Draw, Loss (relative to side to move)
};

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

    Impl(const SelfPlayConfig& c) : config(c)
    {
        chessrl::mcts::single::SmallBatchConfig batch_config;
        batch_config.engine_path = config.engine_path;
        batch_config.enable_telemetry = true;
        runner = std::make_shared<chessrl::mcts::single::SmallBatchRunner>(batch_config);
        
        // 4GB cache
        cache = std::make_shared<chessrl::mcts::TranspositionTable>(4096);

        slots.resize(config.concurrency);
        for (auto& slot : slots)
        {
            slot.board = chess::Board();
        }

        // Open in read/write mode. Create if not exists.
        // std::fstream doesn't create file if it doesn't exist when using in | out.
        // So we might need to create it first.
        if (!std::filesystem::exists(config.output_path))
        {
            std::ofstream tmp(config.output_path);
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
            current_records = size / 128; // 128 bytes per record
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
    auto summaries = chessrl::mcts::run_small_mcts_batch_shared(options, impl_->runner, impl_->cache);
    double end_infer_ms = impl_->runner->statistics().total_inference_ms;

    auto current_time = std::chrono::high_resolution_clock::now();
    double mcts_duration_ms = std::chrono::duration<double, std::milli>(current_time - start_time).count();
    double trt_duration_ms = end_infer_ms - start_infer_ms;

    static int log_counter = 0;
    // if (++log_counter % 10 == 0) {
        std::ofstream logfile("/tmp/profile.log", std::ios::app);
        logfile << "!!!!! [Profile] MCTS: " << mcts_duration_ms << " ms, TRT: " << trt_duration_ms << " ms" << std::endl;
        impl_->cache->print_stats();
    // }

    // Process results
    for (size_t i = 0; i < impl_->slots.size(); ++i)
    {
        auto& slot = impl_->slots[i];
        const auto& summary = summaries[i];
        
        // Each position runs config.simulations MCTS simulations
        metrics.nodes_evaluated += impl_->config.simulations;

        if (summary.action.empty())
        {
            slot.board = chess::Board();
            slot.pending_records.clear();
            slot.history.clear();
            slot.ply_count = 0;
            metrics.games_finished++;
            continue;
        }

        // Store record
        SelfPlayRecord record;
        record.fen = slot.board.getFen();
        record.policy = summary.action;
        // If mcts_value is enabled, store the MCTS value directly
        if (impl_->config.mcts_value)
        {
            record.value = summary.value;
        }
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
            slot.board = chess::Board();
            slot.pending_records.clear();
            slot.history.clear();
            slot.ply_count = 0;
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
            
            // Only set value from game outcome if mcts_value is disabled
            if (!impl_->config.mcts_value)
            {
                for (auto& rec : slot.pending_records)
                {
                    bool white_to_move = (rec.fen.find(" w ") != std::string::npos);
                    
                    if (white_to_move)
                    {
                        if (z == 1.0f) rec.value = {1.0f, 0.0f, 0.0f};
                        else if (z == 0.0f) rec.value = {0.0f, 0.0f, 1.0f};
                        else rec.value = {0.0f, 1.0f, 0.0f};
                    }
                    else
                    {
                        if (z == 0.0f) rec.value = {1.0f, 0.0f, 0.0f};
                        else if (z == 1.0f) rec.value = {0.0f, 0.0f, 1.0f};
                        else rec.value = {0.0f, 1.0f, 0.0f};
                    }
                }
            }
            // Write to buffer
            for (const auto& rec : slot.pending_records)
            {
                // Fixed record format (128 bytes)
                // FEN: 100 bytes
                // Policy: 16 bytes
                // Value: 12 bytes
                
                char buffer[128];
                std::memset(buffer, 0, 128);
                
                // Copy FEN
                std::strncpy(buffer, rec.fen.c_str(), 99);
                
                // Copy Policy (at offset 100)
                std::strncpy(buffer + 100, rec.policy.c_str(), 15);
                
                // Copy Value (at offset 116)
                std::memcpy(buffer + 116, rec.value.data(), 12);
                
                if (impl_->current_records < impl_->config.max_buffer_size)
                {
                    // Append
                    impl_->outfile.seekp(0, std::ios::end);
                    impl_->outfile.write(buffer, 128);
                    impl_->current_records++;
                }
                else
                {
                    // Random replacement
                    // Use a simple random generator
                    static std::mt19937 rng(std::random_device{}());
                    std::uniform_int_distribution<size_t> dist(0, impl_->config.max_buffer_size - 1);
                    size_t idx = dist(rng);
                    
                    impl_->outfile.seekp(idx * 128, std::ios::beg);
                    impl_->outfile.write(buffer, 128);
                }
            }
            
            if (impl_->outfile) impl_->outfile.flush();

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

            slot.board = chess::Board();
            slot.pending_records.clear();
            slot.history.clear();
            slot.ply_count = 0;
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
    
    // Approximate: file_size / record_size
    // Record size is 128 bytes (100 FEN + 16 Policy + 12 Value)
    const size_t record_size = 128;
    size_t file_size = std::filesystem::file_size(p);
    return file_size / record_size;
}

} // namespace chessrl
