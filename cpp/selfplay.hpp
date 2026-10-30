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
    int threads = 0;       // 0 = auto (256 MCTS worker threads)
    int hash_size = 4096;  // MB; transposition-table size shared across slots
    int simulations = 800;
    float cpuct = 0.65f;
    float temperature = 1.3f;
    float temperature_at_root = 1.0f;
    // Virtual-loss batched-descent width. 1 = single-leaf descent;
    // >1 = up to N leaves collected per inference round (bigger GPU batches).
    int virtual_loss_width = 1;
    bool use_dirichlet = true;
    float dirichlet_alpha = 0.12f;
    float dirichlet_epsilon = 0.05f;
    int max_games = -1; // -1 for infinite
    size_t max_buffer_size = 1000000; // Max number of positions in buffer
    bool save_pgn = false;
    std::string pgn_dir = "pgns";
    int random_opening_plies = 0; // 0 = always start from the standard startpos

    // ---- Game and record options. The defaults here switch them off; src/train/config.yaml
    // sets the values the training recipe uses (self-play copies every field). ----
    // Root temperature_at_root applies to the first `temperature_plies` MCTS plies
    // of a game; afterwards `temperature_late` (0 = argmax). -1 = whole game.
    int temperature_plies = -1;
    float temperature_late = 0.0f;
    // Resign when the side to move's root expected score (W + D/2) stays below
    // `resign_threshold` for `resign_consecutive` of its own moves. 0 disables.
    // A `resign_playthrough` fraction of games never resigns; they measure the
    // false-positive rate (would-resign side that did not lose).
    float resign_threshold = 0.0f;
    int resign_consecutive = 2;
    float resign_playthrough = 0.1f;
    int max_plies = 600;
    // Playout-cap randomization (KataGo): each move gets a full search with
    // probability `pcr_full_prob`, else a `pcr_fast_simulations` search with no
    // root noise whose position is not written to the buffer. 1.0 = off.
    float pcr_full_prob = 1.0f;
    int pcr_fast_simulations = 100;
    // Keep the played move's subtree for the game's next search, which then runs only the
    // simulations its root is missing (MctsOptions::reuse_tree).
    bool tree_reuse = false;
    // > 0: while the played move is sampled (the first `temperature_plies`), a cheap
    // move skips its search and samples the raw policy at this temperature (one
    // network evaluation). Sampling the visit counts of a 50-sim search played far
    // worse moves than the raw policy (its visits are much flatter). 0 = off.
    float quick_policy_temperature = 0.0f;
    // true = ring buffer (overwrite oldest); false = random replacement when full.
    bool fifo_buffer = false;
    // Rules-only curriculum: with this probability a game starts from a random
    // legal position with two kings and 1..curriculum_max_pieces other pieces.
    float curriculum_prob = 0.0f;
    int curriculum_max_pieces = 4;
};

// Cumulative self-play statistics, also persisted to `<output_path>.stats.json`.
struct SelfPlayStats
{
    int64_t games = 0;
    int64_t white_wins = 0;
    int64_t black_wins = 0;
    int64_t draws = 0;
    int64_t game_plies = 0;          // sum of MCTS plies over finished games
    int64_t end_checkmate = 0;
    int64_t end_stalemate = 0;
    int64_t end_insufficient = 0;
    int64_t end_fifty = 0;
    int64_t end_repetition = 0;
    int64_t end_max_plies = 0;
    int64_t end_resign = 0;
    int64_t playthrough_games = 0;
    int64_t playthrough_would_resign = 0;   // sides that would have resigned
    int64_t playthrough_false_resign = 0;   // ... and then did not lose
    int64_t curriculum_games = 0;
    int64_t full_searches = 0;
    int64_t fast_searches = 0;
    int64_t policy_moves = 0;  // cheap moves sampled from the raw policy (no search)
    int64_t records_written = 0;
    int64_t engine_swaps = 0;
    int64_t simulations_run = 0;  // simulations searches actually ran
    int64_t reused_visits = 0;    // visits inherited from reused trees (tree_reuse)
};

struct SelfPlayMetrics
{
    int64_t games_finished = 0;
    int64_t positions_generated = 0;
    int64_t nodes_evaluated = 0;
    double runtime_seconds = 0.0;
};

// Snapshot of the inference runner's lifetime totals, plus the process-wide
// encoder timing accumulator. Exposed for the perf bench; not used by the
// trainer.
struct SelfPlayInferenceStats
{
    int64_t batches = 0;
    int64_t positions_in_batches = 0;
    double total_inference_ms = 0.0;
    int64_t encoder_calls = 0;
    double total_encoder_ms = 0.0;
};

class SelfPlayWorker
{
public:
    SelfPlayWorker(const SelfPlayConfig& config);
    ~SelfPlayWorker();

    // Run one batch of MCTS simulations for all boards
    // Returns metrics for this step
    SelfPlayMetrics step();

    // Inspect lifetime inference counters (per-runner + per-encoder).
    SelfPlayInferenceStats inference_stats() const;

    // Get approximate number of positions in buffer
    size_t get_buffer_size() const;

    // Swap the network (.cnet) between steps. Games in progress keep going
    // (rebuilding the worker would discard them and their pending records).
    void set_engine(const std::string& engine_path);

    // Update search / game-policy settings between steps (schedules driven by
    // the Python side). Paths, board count, threads, hash and buffer layout
    // keep their construction-time values.
    void set_config(const SelfPlayConfig& config);

    SelfPlayStats stats() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace chessrl
