#pragma once

#include <string>
#include <vector>
#include <atomic>

namespace chessrl
{

struct SelfPlayConfig
{
    std::string engine_path;
    std::string output_path;
    int concurrency = 256;
    int simulations = 800;
    float cpuct = 0.65f;
    float temperature = 1.3f;
    float temperature_at_root = 1.0f;
    bool use_dirichlet = true;
    float dirichlet_alpha = 0.12f;
    float dirichlet_epsilon = 0.05f;
    int max_games = -1; // -1 for infinite
    size_t max_buffer_size = 1000000; // Max number of positions in buffer
    bool save_pgn = false;
    std::string pgn_dir = "pgns";
    int random_opening_plies = 0; // 0 = always start from the standard startpos
};

struct SelfPlayMetrics
{
    int64_t games_finished = 0;
    int64_t positions_generated = 0;
    int64_t nodes_evaluated = 0;
    double runtime_seconds = 0.0;
};

class SelfPlayWorker
{
public:
    SelfPlayWorker(const SelfPlayConfig& config);
    ~SelfPlayWorker();

    // Run one batch of MCTS simulations for all boards
    // Returns metrics for this step
    SelfPlayMetrics step();
    
    // Get approximate number of positions in buffer
    size_t get_buffer_size() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace chessrl
