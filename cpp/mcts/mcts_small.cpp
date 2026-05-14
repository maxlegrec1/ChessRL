#include "mcts/cache.hpp"
#include "inference.hpp"
#include "mcts/common.hpp"
#include "mcts/small_runner.hpp"
#include "mcts/single_inference.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <unordered_map>
#include <vector>

namespace chessrl::mcts
{
namespace
{

struct MovePrior
{
    chess::Move move;
    int policy_index{-1};
    float prior{0.0F};
    float logit{0.0F};
};

std::string move_to_vocab_key(const chess::Board& board, const chess::Move& move)
{
    std::string uci = chess::uci::moveToUci(move, board.chess960());
    if (!uci.empty() && uci.back() == 'n')
    {
        uci.pop_back();
    }
    return uci;
}

std::unordered_map<std::string, int> build_vocab_index(const std::vector<std::string>& vocab)
{
    std::unordered_map<std::string, int> index;
    index.reserve(vocab.size());
    for (size_t i = 0; i < vocab.size(); ++i)
    {
        index.emplace(vocab[i], static_cast<int>(i));
    }
    return index;
}

// Flat LUT keyed by (from_sq * 64 + to_sq) * 5 + promo. promo: 0=none, 1=Q,
// 2=R, 3=B, 4=N. Knight promos in self-play are stripped to no-promo to mirror
// the existing string-key behavior. -1 means "not in vocab".
//
// Castling moves and chess960 boards fall back to the string path because the
// chess::Move encoding for castling stores the rook square as the destination,
// not the king's final square (which is what the UCI vocab uses).
constexpr size_t kVocabLutSize = 64 * 64 * 5;

const std::array<int32_t, kVocabLutSize>& vocab_lut()
{
    static const std::array<int32_t, kVocabLutSize> lut = []() {
        std::array<int32_t, kVocabLutSize> table;
        table.fill(-1);
        const auto& vocab = small_policy_vocabulary();
        for (size_t i = 0; i < vocab.size(); ++i)
        {
            const auto& s = vocab[i];
            if (s.size() < 4) continue;
            int from_file = s[0] - 'a';
            int from_rank = s[1] - '1';
            int to_file   = s[2] - 'a';
            int to_rank   = s[3] - '1';
            if (from_file < 0 || from_file > 7 || from_rank < 0 || from_rank > 7) continue;
            if (to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) continue;
            int from = from_rank * 8 + from_file;
            int to   = to_rank   * 8 + to_file;
            int promo = 0;
            if (s.size() == 5)
            {
                switch (s[4])
                {
                    case 'q': promo = 1; break;
                    case 'r': promo = 2; break;
                    case 'b': promo = 3; break;
                    case 'n': promo = 4; break;
                    default: continue;
                }
            }
            table[(from * 64 + to) * 5 + promo] = static_cast<int32_t>(i);
        }
        return table;
    }();
    return lut;
}

inline int vocab_index_for_move_fast(const chess::Move& move) noexcept
{
    const int from = static_cast<int>(move.from().index());
    const int to   = static_cast<int>(move.to().index());
    int promo = 0;
    if (move.typeOf() == chess::Move::PROMOTION)
    {
        switch (move.promotionType().internal())
        {
            case chess::PieceType::underlying::QUEEN:  promo = 1; break;
            case chess::PieceType::underlying::ROOK:   promo = 2; break;
            case chess::PieceType::underlying::BISHOP: promo = 3; break;
            case chess::PieceType::underlying::KNIGHT: promo = 0; break;
            default: return -1;
        }
    }
    return vocab_lut()[(from * 64 + to) * 5 + promo];
}

int read_batch_size_env(const char* specific)
{
    if (specific)
    {
        if (const char* value = std::getenv(specific))
        {
            const int numeric = std::atoi(value);
            if (numeric > 0)
            {
                return numeric;
            }
        }
    }
    if (const char* value = std::getenv("CHESSRL_MCTS_BATCH_SIZE"))
    {
        const int numeric = std::atoi(value);
        if (numeric > 0)
        {
            return numeric;
        }
    }
    return 1024;
}

bool telemetry_enabled_env()
{
    if (const char* value = std::getenv("CHESSRL_MCTS_TELEMETRY"))
    {
        std::string lower(value);
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return lower != "0" && lower != "false" && lower != "off";
    }
    return false;
}

single::SmallBatchConfig make_small_batch_config(const MctsOptions& options)
{
    single::SmallBatchConfig cfg;
    if (options.small_engine_path.empty())
    {
        cfg.engine_path = "model.trt";
    }
    else
    {
        cfg.engine_path = options.small_engine_path;
    }
    cfg.max_batch_size = read_batch_size_env("CHESSRL_MCTS_SMALL_BATCH_SIZE");
    cfg.enable_telemetry = telemetry_enabled_env();
    return cfg;
}

} // namespace

class SmallMctsRunner
{
public:
    SmallMctsRunner(MctsOptions options, 
                    std::shared_ptr<single::SmallBatchRunner> shared_runner = {},
                    std::shared_ptr<TranspositionTable> cache = {})
        : options_(std::move(options))
        , vocab_(small_policy_vocabulary())
        , vocab_index_(build_vocab_index(vocab_))
        , small_batch_config_(make_small_batch_config(options_))
        , small_batch_runner_(
              shared_runner ? std::move(shared_runner)
                            : std::make_shared<single::SmallBatchRunner>(small_batch_config_))
        , cache_(std::move(cache))
        , rng_(std::random_device{}())
    {
        // Phase B: Store root board for state reconstruction during traversal
        root_board_ = apply_history(options_.request);
        tree_.add_root(root_board_, options_.request.history);
        // Build the initial AlphaZero-style 8-board history stack by
        // replaying the move history. Slots 1..7 capture positions t-1..t-7
        // before reaching the root. Used as the per-simulation seed.
        root_history_stack_ = build_root_history_stack(
            options_.request.fen, options_.request.history, root_board_);
        const size_t expected_nodes = 1 + static_cast<size_t>(options_.simulations) * 2;
        tree_.reserve(expected_nodes);
        path_buffer_.reserve(64);
    }

    MctsSummary run()
    {
        // Phase B: Pass root board + initial history stack to evaluate_node.
        evaluate_node(0, root_board_, root_history_stack_);

        for (int i = 0; i < options_.simulations; ++i)
        {
            run_simulation();
        }

        refine_wdl_recursive();

        MctsSummary summary = summarize();
        
        if (small_batch_config_.enable_telemetry)
        {
            auto stats = small_batch_runner_->statistics();
            if (stats.batches > 0)
            {
                std::clog << "[telemetry][small] batches=" << stats.batches
                          << " positions=" << stats.positions
                          << " infer_ms=" << stats.total_inference_ms << '\n';
            }
        }
        return summary;
    }

private:
    void run_simulation()
    {
        path_buffer_.clear();
        path_buffer_.push_back(0);
        std::vector<int>& path = path_buffer_;
        int node_index = 0;

        // Phase B: Maintain local board state, initialized to root at start of simulation.
        chess::Board current_board = root_board_;
        // Per-simulation history ring. Starts as the root stack (which
        // captures the real-game history t-7..t-1 + current). Rotated and
        // refreshed on every makeMove during tree descent. Lives on the
        // stack ⇒ inherently per-thread.
        HistoryStack history_stack = root_history_stack_;

        while (true)
        {
            TreeNode& node = tree_.nodes[node_index];

            if (node.terminal)
            {
                const float value = terminal_value(current_board).value_or(node.value_prior);
                const std::array<float, 3> leaf_wdl = wdl_from_terminal(value);
                detail::backpropagate_full(tree_, path, value, leaf_wdl, /*leaf_plies=*/0.0F);
                return;
            }

            if (!node.expanded || node.children.empty())
            {
                evaluate_node(node_index, current_board, history_stack);
                const TreeNode& evaluated = tree_.nodes[node_index];
                detail::backpropagate_full(
                    tree_, path, evaluated.value_prior, evaluated.wdl, evaluated.leaf_plies);
                return;
            }

            const int child_index = select_child(node_index);
            if (child_index < 0)
            {
                evaluate_node(node_index, current_board, history_stack);
                const TreeNode& evaluated = tree_.nodes[node_index];
                detail::backpropagate_full(
                    tree_, path, evaluated.value_prior, evaluated.wdl, evaluated.leaf_plies);
                return;
            }

            // Phase B: Apply the move to our local board tracker and shift
            // the history ring before descending.
            current_board.makeMove(tree_.nodes[child_index].move_from_parent);
            PiecePlanes snap = snapshot_piece_planes(current_board);
            snap.repetition = current_board.isRepetition(1);
            history_stack_push(history_stack, snap);
            node_index = child_index;
            path.push_back(node_index);
        }
    }


    int select_child(int node_index) const
    {
        const TreeNode& node = tree_.nodes[node_index];
        if (node.children.empty())
        {
            return -1;
        }

        const float parent_visits = static_cast<float>(std::max(1, node.visit_count));
        const float sqrt_visits = std::sqrt(parent_visits);

        int best_child = -1;
        float best_score = -std::numeric_limits<float>::infinity();

        // Q + U + ε * (1 - 2 * W_parent_pov) * tanh(plies_avg / 25)
        // - W_parent_pov is child's L (from child's POV), since flipping a
        //   child's perspective into the parent's swaps W↔L.
        // - plies_avg is the in-tree backed-up plies-remaining at the child
        //   (mlh_q-seeded leaves + per-ply +1 increments via backup). We
        //   squash with tanh to bound the modifier to [-ε, ε] regardless of
        //   game length — long-horizon positions saturate.
        // - sign of (1 - 2W): when winning (W large), modifier penalizes high
        //   MLH (prefer faster). When losing/drawing, modifier rewards high
        //   MLH (prefer slower / drag it out).
        constexpr float kMlhEpsilon = 0.05F;
        constexpr float kPliesScale = 1.0F / 25.0F;

        for (size_t i = 0; i < node.children.size(); ++i)
        {
            const int child_index = node.children[i];
            const TreeNode& child = tree_.nodes[child_index];
            const float prior = node.child_priors[i];
            const float child_mean = child.visit_count > 0 ? child.value_sum / static_cast<float>(child.visit_count) : 0.0F;
            const float q = -child_mean;
            const float u = options_.cpuct * prior * sqrt_visits / (1.0F + static_cast<float>(child.visit_count));

            float mlh_modifier = 0.0F;
            if (child.visit_count > 0)
            {
                // child.wdl is in child's perspective; flip to parent's POV.
                const float w_parent_pov = child.wdl[2];
                const float inv_n = 1.0F / static_cast<float>(child.visit_count);
                const float plies_avg = child.plies_sum * inv_n;
                mlh_modifier = (1.0F - 2.0F * w_parent_pov) * std::tanh(plies_avg * kPliesScale);
            }

            const float score = q + u + kMlhEpsilon * mlh_modifier;
            if (score > best_score)
            {
                best_score = score;
                best_child = child_index;
            }
        }

        return best_child;
    }

    float evaluate_node(
        int node_index,
        const chess::Board& board_state,
        const HistoryStack& history)
    {
        TreeNode& node = tree_.nodes[node_index];
        if (node.expanded && !node.children.empty())
        {
            return node.value_prior;
        }

        // Phase B: Use passed board_state instead of node.board
        if (auto terminal = terminal_value(board_state))
        {
            node.expanded = true;
            node.terminal = true;
            node.value_prior = *terminal;
            node.wdl = wdl_from_terminal(*terminal);
            node.leaf_plies = 0.0F;
            return node.value_prior;
        }

        // Phase B: Use passed board_state for encoding
        // Cache key includes the history-stack contents because the NN's
        // output now depends on the full 8-board stack, not just the
        // current position. Two visits to the same zobrist with different
        // surrounding history MUST NOT share an entry.
        SmallPolicyValue eval;
        bool cache_hit = false;
        uint64_t hash = compute_cache_key(board_state.zobrist(), history);

        if (cache_)
        {
            if (auto cached = cache_->get(hash))
            {
                eval = *cached;
                cache_hit = true;
            }
        }

        if (!cache_hit)
        {
            auto encoded = acquire_small_encoding(history, board_state);
            auto ticket = small_batch_runner_->enqueue(*encoded);
            eval = ticket.get();

            if (cache_)
            {
                cache_->put(hash, eval);
            }
        }

        if (eval.policy_logits.size() != chessrl::small_policy_size())
        {
            throw std::runtime_error("Single-batch small inference returned unexpected policy size.");
        }

        // WDL is already in side-to-move perspective: [win, draw, loss]
        node.wdl = eval.wdl;
        node.value_prior = node.wdl[0] - node.wdl[2];
        // mlh_q is the search-aware head, output as plies/25. Convert to raw
        // plies for the +depth backup math. Clamp at 0 so a negative-prediction
        // outlier doesn't pollute the running plies-remaining estimate.
        node.leaf_plies = std::max(0.0F, eval.mlh_q * 25.0F);

        // Phase B: Use passed board_state for computing priors
        std::vector<MovePrior> priors = compute_priors(board_state, eval);
        if (priors.empty())
        {
            node.terminal = true;
            node.expanded = true;
            return node.value_prior;
        }

        if (node.parent == -1 && options_.use_dirichlet)
        {
            apply_dirichlet_noise(priors);
        }

        // Phase B: Use passed board_state as parent board
        const chess::Board parent_board = board_state;
        const float value_prior = node.value_prior;
        for (const auto& entry : priors)
        {
            chess::Board child_board = parent_board;
            child_board.makeMove(entry.move);
            tree_.add_child(node_index, entry.move, entry.prior, child_board);
        }

        TreeNode& refreshed = tree_.nodes[node_index];
        refreshed.expanded = true;

        return value_prior;
    }


    std::vector<MovePrior> compute_priors(const chess::Board& board, const SmallPolicyValue& eval) const
    {
        chess::Movelist legal_moves;
        chess::movegen::legalmoves(legal_moves, board);

        std::vector<MovePrior> entries;
        entries.reserve(legal_moves.size());

        float max_logit = -std::numeric_limits<float>::infinity();
        bool has_logit = false;
        int mapped_count = 0;

        const bool use_fast_lut = !board.chess960();
        for (const auto& move : legal_moves)
        {
            MovePrior entry;
            entry.move = move;
            entry.logit = -std::numeric_limits<float>::infinity();

            // Fast path: O(1) array lookup keyed on Move bits. Falls back to
            // the string-keyed map for chess960 (different castling UCI) and
            // for any move the LUT couldn't resolve directly (castling under
            // standard rules — chess::Move stores the rook square, not the
            // king's destination, so the LUT misses it).
            int policy_index = -1;
            if (use_fast_lut && move.typeOf() != chess::Move::CASTLING)
            {
                policy_index = vocab_index_for_move_fast(move);
            }
            if (policy_index < 0)
            {
                const std::string key = move_to_vocab_key(board, move);
                auto it = vocab_index_.find(key);
                if (it != vocab_index_.end()) policy_index = it->second;
            }
            if (policy_index >= 0)
            {
                entry.policy_index = policy_index;
                if (entry.policy_index < static_cast<int>(eval.policy_logits.size())) {
                    entry.logit = eval.policy_logits[static_cast<size_t>(entry.policy_index)];
                    if (!has_logit || entry.logit > max_logit)
                    {
                        max_logit = entry.logit;
                        has_logit = true;
                    }
                    ++mapped_count;
                }
                else
                {
                    entry.policy_index = -1;
                }
            }

            entries.push_back(std::move(entry));
        }

        if (entries.empty())
        {
            return entries;
        }

        if (mapped_count == 0)
        {
            const float uniform = 1.0F / static_cast<float>(entries.size());
            for (auto& entry : entries)
            {
                entry.prior = uniform;
            }
            return entries;
        }

        if (options_.temperature <= 1e-5F)
        {
            auto best_it = std::max_element(entries.begin(), entries.end(), [](const MovePrior& a, const MovePrior& b) {
                return a.logit < b.logit;
            });
            for (auto& entry : entries)
            {
                const bool is_best = (entry.policy_index >= 0) && (&entry == &*best_it);
                entry.prior = is_best ? 1.0F : 0.0F;
            }
            return entries;
        }

        const float inv_temp = 1.0F / options_.temperature;
        double sum = 0.0;
        for (auto& entry : entries)
        {
            if (entry.policy_index < 0)
            {
                entry.prior = 0.0F;
                continue;
            }
            const double scaled = std::exp(static_cast<double>((entry.logit - max_logit) * inv_temp));
            if (!std::isfinite(scaled))
            {
                entry.prior = 0.0F;
                continue;
            }
            entry.prior = static_cast<float>(scaled);
            sum += scaled;
        }

        if (sum <= 0.0)
        {
            const float uniform = 1.0F / static_cast<float>(mapped_count);
            for (auto& entry : entries)
            {
                if (entry.policy_index >= 0)
                {
                    entry.prior = uniform;
                }
            }
            return entries;
        }

        const float inv_sum = static_cast<float>(1.0 / sum);
        for (auto& entry : entries)
        {
            if (entry.policy_index >= 0)
            {
                entry.prior *= inv_sum;
            }
        }

        return entries;
    }

    void apply_dirichlet_noise(std::vector<MovePrior>& priors)
    {
        if (priors.empty())
        {
            return;
        }

        std::gamma_distribution<float> gamma(options_.dirichlet_alpha, 1.0F);
        dirichlet_noise_.resize(priors.size());
        float sum = 0.0F;
        for (float& n : dirichlet_noise_)
        {
            n = gamma(rng_);
            sum += n;
        }
        if (sum <= 0.0F)
        {
            return;
        }

        const float epsilon = options_.dirichlet_epsilon;
        for (size_t i = 0; i < priors.size(); ++i)
        {
            const float dirichlet = dirichlet_noise_[i] / sum;
            priors[i].prior = (1.0F - epsilon) * priors[i].prior + epsilon * dirichlet;
        }
    }

    void refine_wdl_recursive()
    {
        std::vector<bool> visited(tree_.nodes.size(), false);
        detail::refine_wdl_node(tree_, 0, visited);
    }

    MctsSummary summarize()
    {
        const TreeNode& root = tree_.nodes.front();

        std::vector<std::pair<std::string, float>> distribution;
        distribution.reserve(root.children.size());

        struct Aggregated
        {
            std::string uci;
            int visit_sum{0};
            float prior_sum{0.0F};
            float value_sum{0.0F};
            int representative_child{-1};
        };

        std::vector<Aggregated> aggregated;
        aggregated.reserve(root.children.size());
        std::unordered_map<std::string, size_t> index;

        int total_visits = 0;
        float total_priors = 0.0F;

        for (size_t i = 0; i < root.children.size(); ++i)
        {
            const int child_index = root.children[i];
            const TreeNode& child = tree_.nodes[child_index];
            const chess::Move move = root.child_moves[i];
            // Phase B: Use root_board_ instead of root.board
            const std::string uci = chess::uci::moveToUci(move, root_board_.chess960());

            auto [it, inserted] = index.emplace(uci, aggregated.size());
            if (inserted)
            {
                aggregated.push_back(Aggregated{uci});
            }

            Aggregated& entry = aggregated[it->second];
            entry.visit_sum += child.visit_count;
            entry.prior_sum += root.child_priors[i];
            entry.value_sum += child.value_sum;
            if (entry.representative_child < 0 || child.visit_count > tree_.nodes[entry.representative_child].visit_count)
            {
                entry.representative_child = child_index;
            }

            total_visits += child.visit_count;
            total_priors += root.child_priors[i];
        }

        if (aggregated.empty())
        {
            return MctsSummary{
                .action = {},
                .policy = {},
                .value = root.wdl,
                .rollouts = {},
            };
        }

        const float visit_sum = static_cast<float>(total_visits);
        const float prior_total = total_priors;

        if (visit_sum <= 0.0F && prior_total <= 0.0F)
        {
            const float uniform = 1.0F / static_cast<float>(aggregated.size());
            for (const auto& entry : aggregated)
            {
                distribution.emplace_back(entry.uci, uniform);
            }
        }
        else
        {
            const float normalizer = visit_sum > 0.0F ? visit_sum : prior_total;
            for (const auto& entry : aggregated)
            {
                const float value = visit_sum > 0.0F ? static_cast<float>(entry.visit_sum) : entry.prior_sum;
                distribution.emplace_back(entry.uci, value / normalizer);
            }
        }

        std::vector<std::pair<std::string, float>> sorted = distribution;
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            if (a.second == b.second)
            {
                return a.first < b.first;
            }
            return a.second > b.second;
        });

        auto select_argmax = [&]() -> std::string {
            size_t best_index = 0;
            float best_score = -1.0F;
            if (visit_sum > 0.0F)
            {
                for (const auto& entry : aggregated)
                {
                    const float visits = static_cast<float>(entry.visit_sum);
                    if (visits > best_score)
                    {
                        best_score = visits;
                        best_index = index.at(entry.uci);
                    }
                }
            }
            else
            {
                for (const auto& entry : aggregated)
                {
                    if (entry.prior_sum > best_score)
                    {
                        best_score = entry.prior_sum;
                        best_index = index.at(entry.uci);
                    }
                }
            }
            return aggregated[best_index].uci;
        };

        auto sample_action = [&]() -> std::optional<std::string> {
            if (distribution.empty())
            {
                return std::nullopt;
            }

            std::vector<double> weights(distribution.size(), 0.0);
            const double inv_temp = 1.0 / static_cast<double>(options_.temperature_at_root);
            double total_weight = 0.0;

            for (size_t i = 0; i < distribution.size(); ++i)
            {
                const double prob = std::max(0.0, static_cast<double>(distribution[i].second));
                if (prob <= 0.0)
                {
                    continue;
                }

                double weight = options_.temperature_at_root == 1.0F ? prob : std::pow(prob, inv_temp);
                if (!std::isfinite(weight) || weight <= 0.0)
                {
                    continue;
                }

                weights[i] = weight;
                total_weight += weight;
            }

            if (total_weight <= 0.0)
            {
                return std::nullopt;
            }

            std::uniform_real_distribution<double> dist(0.0, total_weight);
            double draw = dist(rng_);
            for (size_t i = 0; i < weights.size(); ++i)
            {
                const double weight = weights[i];
                if (weight <= 0.0)
                {
                    continue;
                }
                draw -= weight;
                if (draw <= 0.0)
                {
                    return distribution[i].first;
                }
            }

            return distribution.back().first;
        };

        std::string action;
        if (options_.temperature_at_root <= 1e-5F)
        {
            action = select_argmax();
        }
        else
        {
            action = sample_action().value_or(select_argmax());
        }

        MctsSummary summary;
        summary.action = action;
        summary.policy = std::move(sorted);
        summary.value = root.wdl;
        // Root's visit-weighted backed plies. The root itself has been visited
        // simulations + 1 times in the current backup scheme (initial expansion
        // contributes one visit too), so this is a valid average. When visits
        // == 0 (degenerate), keep the negative sentinel.
        summary.backed_plies = root.visit_count > 0
            ? root.plies_sum / static_cast<float>(root.visit_count)
            : -1.0F;
        summary.rollouts.reserve(aggregated.size());
        for (const auto& entry : aggregated)
        {
            RolloutDetail detail;
            detail.root_move = entry.uci;
            detail.visit_count = entry.visit_sum;
            if (visit_sum > 0.0F)
            {
                detail.visit_fraction = static_cast<float>(entry.visit_sum) / visit_sum;
            }
            detail.prior = entry.prior_sum;
            detail.average_value = entry.visit_sum > 0 ? -entry.value_sum / static_cast<float>(entry.visit_sum) : 0.0F;
            if (entry.representative_child >= 0)
            {
                // Phase B: Pass root_board_ to principal_variation
                detail.continuation = principal_variation(tree_, entry.representative_child, root_board_);
            }
            summary.rollouts.push_back(std::move(detail));
        }
        return summary;
    }

    std::shared_ptr<const single::SmallEncodedPosition> acquire_small_encoding(
        const HistoryStack& history,
        const chess::Board& board)
    {
        // Each leaf in MCTS is by definition unvisited; the prior encoding cache
        // (keyed on HistoryEntry*) had near-zero hit rate but locked a mutex on
        // every leaf, serializing the worker threads of a single board. Drop it.
        return std::make_shared<single::SmallEncodedPosition>(
            single::encode_small_position(board, history));
    }

    // Fold the history stack into a 64-bit hash by combining each slot's
    // bitboards with a Murmur-style mixer. Used to ensure the TT cache
    // doesn't return stale evaluations when two visits to the same physical
    // position have different surrounding histories.
    static std::uint64_t compute_cache_key(std::uint64_t zobrist, const HistoryStack& history)
    {
        auto mix = [](std::uint64_t h) {
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33;
            return h;
        };
        std::uint64_t h = zobrist;
        for (int slot = 0; slot < kHistoryDepth; ++slot)
        {
            const auto& p = history[static_cast<size_t>(slot)];
            for (int k = 0; k < 12; ++k)
            {
                h = mix(h ^ (p.bitboards[static_cast<size_t>(k)]
                             + static_cast<std::uint64_t>(slot * 31 + k)));
            }
            if (p.repetition)
            {
                h = mix(h ^ (0x9e3779b97f4a7c15ULL + static_cast<std::uint64_t>(slot)));
            }
        }
        return h;
    }

    MctsOptions options_;
    SearchTree tree_;
    chess::Board root_board_; // Phase B: Store root board for state reconstruction
    HistoryStack root_history_stack_;  // Initial 8-slot piece-plane stack at root.
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> vocab_index_;
    single::SmallBatchConfig small_batch_config_;
    std::shared_ptr<single::SmallBatchRunner> small_batch_runner_;
    std::shared_ptr<TranspositionTable> cache_;
    std::mt19937 rng_;
    std::vector<int> path_buffer_;
    std::vector<float> dirichlet_noise_;
};

MctsSummary run_small_mcts(MctsOptions options)
{
    SmallMctsRunner runner(std::move(options));
    return runner.run();
}

std::vector<MctsSummary> run_small_mcts_batch_shared(
    std::vector<MctsOptions> batch_options,
    std::shared_ptr<single::SmallBatchRunner> shared_runner,
    std::shared_ptr<TranspositionTable> cache,
    int requested_threads)
{
    if (batch_options.empty())
    {
        return {};
    }

    std::vector<MctsSummary> summaries(batch_options.size());

    // Worker count: caller picks via config.threads. 0 = auto (legacy 256 cap).
    // No point spawning more workers than there are jobs.
    size_t num_threads = (requested_threads > 0)
        ? static_cast<size_t>(requested_threads)
        : size_t{256};
    num_threads = std::min(num_threads, batch_options.size());
    std::atomic<size_t> next_job_index{0};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    // Worker routine: fetch jobs from the queue until all are processed
    auto worker_routine = [&batch_options, shared_runner, cache, &summaries, &next_job_index]() {
        while (true)
        {
            const size_t job_id = next_job_index.fetch_add(1);
            if (job_id >= batch_options.size())
            {
                break;
            }
            
            SmallMctsRunner runner(batch_options[job_id], shared_runner, cache);
            summaries[job_id] = runner.run();
        }
    };

    // Spawn only num_threads workers
    for (size_t i = 0; i < num_threads; ++i)
    {
        workers.emplace_back(worker_routine);
    }

    // Join all workers
    for (auto& worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }

    return summaries;
}

std::vector<MctsSummary> run_small_mcts_batch(std::vector<MctsOptions> batch_options)
{
    if (batch_options.empty())
    {
        return {};
    }

    single::SmallBatchConfig shared_config = make_small_batch_config(batch_options.front());
    for (size_t i = 1; i < batch_options.size(); ++i)
    {
        const auto cfg = make_small_batch_config(batch_options[i]);
        if (cfg.engine_path != shared_config.engine_path)
        {
            throw std::invalid_argument("All small MCTS batch requests must use the same engine path.");
        }
    }

    auto shared_runner = std::make_shared<single::SmallBatchRunner>(shared_config);
    return run_small_mcts_batch_shared(std::move(batch_options), shared_runner);
}

#ifndef CHESSRL_MCTS_NO_STANDALONE
int main(int argc, char** argv)
{
    try
    {
        auto options = chessrl::mcts::parse_cli(argc, argv);
        auto summary = chessrl::mcts::run_small_mcts(std::move(options));

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\n";
        if (summary.action.empty())
        {
            std::cout << "  \"action\": null,\n";
        }
        else
        {
            std::cout << "  \"action\": \"" << summary.action << "\",\n";
        }
        std::cout << "  \"policy\": {\n";
        for (size_t i = 0; i < summary.policy.size(); ++i)
        {
            const auto& [move, prob] = summary.policy[i];
            std::cout << "    \"" << move << "\": " << prob;
            if (i + 1 < summary.policy.size())
            {
                std::cout << ",";
            }
            std::cout << "\n";
        }
        std::cout << "  },\n";
        std::cout << "  \"value\": [" << summary.value[0] << ", " << summary.value[1] << ", " << summary.value[2] << "]\n";
        std::cout << "}\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
#endif // CHESSRL_MCTS_NO_STANDALONE

} // namespace chessrl::mcts