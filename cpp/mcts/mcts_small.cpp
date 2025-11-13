#include "inference.hpp"
#include "mcts/common.hpp"
#include "mcts/small_runner.hpp"
#include "mcts/single_inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
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
    std::string uci;
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

std::optional<float> terminal_value(const chess::Board& board)
{
    const auto result = board.isGameOver().second;
    switch (result)
    {
    case chess::GameResult::WIN:
        return 1.0F;
    case chess::GameResult::LOSE:
        return -1.0F;
    case chess::GameResult::DRAW:
        return 0.0F;
    default:
        return std::nullopt;
    }
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

} // namespace

class SmallMctsRunner
{
public:
    explicit SmallMctsRunner(MctsOptions options)
        : options_(std::move(options))
        , vocab_(small_policy_vocabulary())
        , vocab_index_(build_vocab_index(vocab_))
        , rng_(std::random_device{}())
    {
        const chess::Board board = apply_history(options_.request);
        tree_.add_root(board, options_.request.history);
    }

    MctsSummary run()
    {
        evaluate_node(0);
        for (int i = 0; i < options_.simulations; ++i)
        {
            run_simulation();
        }
        return summarize();
    }

private:
    void run_simulation()
    {
        std::vector<int> path;
        path.reserve(64);
        int node_index = 0;
        path.push_back(node_index);

        while (true)
        {
            TreeNode& node = tree_.nodes[node_index];

            if (node.terminal)
            {
                float value = terminal_value(node.board).value_or(node.value_prior);
                backpropagate(path, value);
                return;
            }

            if (!node.expanded || node.children.empty())
            {
                float value = evaluate_node(node_index);
                backpropagate(path, value);
                return;
            }

            const int child_index = select_child(node_index);
            if (child_index < 0)
            {
                float value = evaluate_node(node_index);
                backpropagate(path, value);
                return;
            }

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

        for (size_t i = 0; i < node.children.size(); ++i)
        {
            const int child_index = node.children[i];
            const TreeNode& child = tree_.nodes[child_index];
            const float prior = node.child_priors[i];
            const float q = child.visit_count > 0 ? child.value_sum / static_cast<float>(child.visit_count) : 0.0F;
            const float u = options_.cpuct * prior * sqrt_visits / (1.0F + static_cast<float>(child.visit_count));
            const float score = q + u;
            if (score > best_score)
            {
                best_score = score;
                best_child = child_index;
            }
        }

        return best_child;
    }

    void backpropagate(const std::vector<int>& path, float value)
    {
        float current = value;
        for (auto it = path.rbegin(); it != path.rend(); ++it)
        {
            TreeNode& node = tree_.nodes[*it];
            node.visit_count += 1;
            node.value_sum += current;
            current = -current;
        }
    }

    float evaluate_node(int node_index)
    {
        TreeNode& node = tree_.nodes[node_index];
        if (node.expanded && !node.children.empty())
        {
            return node.value_prior;
        }

        if (auto terminal = terminal_value(node.board))
        {
            node.expanded = true;
            node.value_prior = *terminal;
            if (*terminal > 0.0F)
            {
                node.wdl = {1.0F, 0.0F, 0.0F};
            }
            else if (*terminal < 0.0F)
            {
                node.wdl = {0.0F, 0.0F, 1.0F};
            }
            else
            {
                node.wdl = {0.0F, 1.0F, 0.0F};
            }
            return node.value_prior;
        }

        const std::string fen = node.board.getFen();
        SmallPolicyValue eval = single::evaluate_small_position(fen, options_.small_engine_path);
        if (eval.policy_logits.size() != chessrl::small_policy_size())
        {
            throw std::runtime_error("Single-batch small inference returned unexpected policy size.");
        }

        node.wdl = orient_small_wdl(node.board, eval.wdl);
        node.value_prior = node.wdl[0] - node.wdl[2];

        std::vector<MovePrior> priors = compute_priors(node.board, eval);
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

        const chess::Board parent_board = node.board;
        for (const auto& entry : priors)
        {
            chess::Board child_board = parent_board;
            child_board.makeMove(entry.move);
            tree_.add_child(node_index, entry.move, entry.prior, child_board, entry.uci);
        }

        node.expanded = true;

        return node.value_prior;
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

        for (const auto& move : legal_moves)
        {
            MovePrior entry;
            entry.move = move;
            entry.uci = chess::uci::moveToUci(move, board.chess960());
            entry.logit = -std::numeric_limits<float>::infinity();

            const std::string key = move_to_vocab_key(board, move);
            auto it = vocab_index_.find(key);
            if (it != vocab_index_.end())
            {
                entry.policy_index = it->second;
                if (entry.policy_index >= 0 && entry.policy_index < static_cast<int>(eval.policy_logits.size()))
                {
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
        std::vector<float> noise(priors.size());
        float sum = 0.0F;
        for (float& n : noise)
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
            const float dirichlet = noise[i] / sum;
            priors[i].prior = (1.0F - epsilon) * priors[i].prior + epsilon * dirichlet;
        }
    }

    MctsSummary summarize() const
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
            const std::string uci = chess::uci::moveToUci(move, root.board.chess960());

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

        const std::string action = aggregated[best_index].uci;

        MctsSummary summary;
        summary.action = action;
        summary.policy = std::move(sorted);
        summary.value = root.wdl;
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
            detail.average_value = entry.visit_sum > 0 ? entry.value_sum / static_cast<float>(entry.visit_sum) : 0.0F;
            if (entry.representative_child >= 0)
            {
                detail.continuation = principal_variation(tree_, entry.representative_child);
            }
            summary.rollouts.push_back(std::move(detail));
        }
        return summary;
    }

    MctsOptions options_;
    SearchTree tree_;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> vocab_index_;
    std::mt19937 rng_;
};

MctsSummary run_small_mcts(MctsOptions options)
{
    SmallMctsRunner runner(std::move(options));
    return runner.run();
}

#ifndef CHESSRL_MCTS_NO_STANDALONE
} // namespace chessrl::mcts

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