#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>

#include "chess-library/include/chess.hpp"
#include "inference.hpp"
#include "mcts/common.hpp"
#include "mcts/single_inference.hpp"
#include "mcts/small_runner.hpp"
#include "mcts/summary.hpp"
#include "selfplay.hpp"

namespace py = pybind11;

namespace
{

py::dict summary_to_python(const chessrl::mcts::MctsSummary& summary)
{
    py::dict result;
    result["action"] = summary.action;
    result["policy"] = summary.policy;
    result["value"] = summary.value;

    py::dict q_values;
    for (const auto& detail : summary.rollouts)
    {
        q_values[py::str(detail.root_move)] = detail.average_value;
    }
    result["q_values"] = std::move(q_values);

    return result;
}

} // namespace

PYBIND11_MODULE(_inference_cpp, m)
{
    py::class_<chessrl::InferenceTimings>(m, "InferenceTimings")
        .def_readonly("encode_ms", &chessrl::InferenceTimings::encodeMs)
        .def_readonly("inference_ms", &chessrl::InferenceTimings::inferMs)
        .def_readonly("sampling_ms", &chessrl::InferenceTimings::sampleMs)
        .def_readonly("total_ms", &chessrl::InferenceTimings::totalMs);

    py::class_<chessrl::SmallInferenceResult>(m, "SmallInferenceResult")
        .def_property_readonly("moves", [](const chessrl::SmallInferenceResult& self) { return self.moves; })
        .def_property_readonly("wdl", [](const chessrl::SmallInferenceResult& self) { return self.wdl; })
        .def_property_readonly("timings", [](const chessrl::SmallInferenceResult& self) { return self.timings; });

    // Encoder parity hook: returns the C++-side input plane vector for a
    // (FEN, history) pair so the Python parity test can bit-compare it
    // against fen_encoder.fen_to_tensor. Pure encoding; no NN call.
    m.def(
        "encode_position_with_history",
        [](const std::string& fen, const std::vector<std::string>& moves) {
            // Build the root board from `fen` and replay `moves` on top.
            const std::string starting_fen = fen.empty()
                ? std::string(chess::constants::STARTPOS)
                : fen;
            chess::Board board(starting_fen);
            for (const auto& mv_uci : moves)
            {
                chess::Move mv = chess::uci::uciToMove(board, mv_uci);
                if (mv == chess::Move::NO_MOVE)
                {
                    throw std::runtime_error(
                        "encode_position_with_history: failed to parse move " + mv_uci);
                }
                board.makeMove(mv);
            }
            // Build the 8-board history stack the same way the runtime does.
            chessrl::mcts::HistoryStack stack =
                chessrl::mcts::build_root_history_stack(starting_fen, moves, board);
            auto encoded = chessrl::mcts::single::encode_small_position(board, stack);
            return encoded.features;
        },
        py::arg("fen"),
        py::arg("moves"),
        "Return the C++ encoder's 8x8x111 input plane vector for a FEN + UCI history.");

    m.def(
        "small_generate_moves",
        [](const std::vector<std::string>& fens, float temperature, int iterations, const std::string& engine_path) {
            return chessrl::small_generate_moves(fens, temperature, iterations, engine_path);
        },
        py::arg("fens"),
        py::arg("temperature"),
        py::arg("iterations") = 1,
        py::arg("engine_path") = std::string("model.trt"),
        "Run the TensorRT policy head and return sampled moves with timing stats.");

    m.def(
        "small_mcts_search_batch",
        [](const std::vector<std::string>& fens,
           int simulations,
           float cpuct,
           float temperature,
           const std::string& engine_path,
           float temperature_at_root,
           bool use_dirichlet,
           float dirichlet_alpha,
           float dirichlet_epsilon) {
            std::vector<chessrl::mcts::MctsOptions> requests;
            requests.reserve(fens.size());
            for (const auto& fen : fens)
            {
                chessrl::mcts::MctsOptions options;
                options.request.fen = fen;
                options.simulations = simulations;
                options.cpuct = cpuct;
                options.temperature = temperature;
                options.small_engine_path = engine_path;
                options.temperature_at_root = temperature_at_root;
                options.use_dirichlet = use_dirichlet;
                options.dirichlet_alpha = dirichlet_alpha;
                options.dirichlet_epsilon = dirichlet_epsilon;
                requests.push_back(std::move(options));
            }

            auto summaries = chessrl::mcts::run_small_mcts_batch(std::move(requests));
            std::vector<py::dict> python_results;
            python_results.reserve(summaries.size());
            for (const auto& summary : summaries)
            {
                python_results.push_back(summary_to_python(summary));
            }
            return python_results;
        },
        py::arg("fens"),
        py::arg("simulations") = 800,
        py::arg("cpuct") = 1.5F,
        py::arg("temperature") = 1.0F,
        py::arg("engine_path") = std::string("model.trt"),
        py::arg("temperature_at_root") = 0.0F,
        py::arg("use_dirichlet") = false,
        py::arg("dirichlet_alpha") = 0.3F,
        py::arg("dirichlet_epsilon") = 0.25F,
        "Execute small-model MCTS searches for a batch of root FENs and return per-root action, policy, and Q-values.");

    // History-based entry point to avoid reconstructing FEN strings in Python.
    m.def(
        "small_mcts_search_batch_with_history",
        [](const std::vector<std::vector<std::string>>& histories,
           const std::vector<std::string>& fens,
           int simulations,
           float cpuct,
           float temperature,
           const std::string& engine_path,
           float temperature_at_root,
           bool use_dirichlet,
           float dirichlet_alpha,
           float dirichlet_epsilon) {
            std::vector<chessrl::mcts::MctsOptions> requests;
            requests.reserve(histories.size());
            for (size_t i = 0; i < histories.size(); ++i)
            {
                const std::vector<std::string>& history = histories[i];
                const std::string fen = (i < fens.size()) ? fens[i] : std::string();
                chessrl::mcts::MctsOptions options;
                options.request.fen = fen;
                options.request.history = history;
                options.simulations = simulations;
                options.cpuct = cpuct;
                options.temperature = temperature;
                options.small_engine_path = engine_path;
                options.temperature_at_root = temperature_at_root;
                options.use_dirichlet = use_dirichlet;
                options.dirichlet_alpha = dirichlet_alpha;
                options.dirichlet_epsilon = dirichlet_epsilon;
                requests.push_back(std::move(options));
            }

            auto summaries = chessrl::mcts::run_small_mcts_batch(std::move(requests));
            std::vector<py::dict> python_results;
            python_results.reserve(summaries.size());
            for (const auto& summary : summaries)
            {
                python_results.push_back(summary_to_python(summary));
            }
            return python_results;
        },
        py::arg("histories"),
        py::arg("fens") = std::vector<std::string>(),
        py::arg("simulations") = 800,
        py::arg("cpuct") = 1.5F,
        py::arg("temperature") = 1.0F,
        py::arg("engine_path") = std::string("model.trt"),
        py::arg("temperature_at_root") = 0.0F,
        py::arg("use_dirichlet") = false,
        py::arg("dirichlet_alpha") = 0.3F,
        py::arg("dirichlet_epsilon") = 0.25F,
        "Execute small-model MCTS searches for a batch of move histories and return per-root action, policy, and Q-values.");

    py::class_<chessrl::SelfPlayMetrics>(m, "SelfPlayMetrics")
        .def_readonly("games_finished", &chessrl::SelfPlayMetrics::games_finished)
        .def_readonly("positions_generated", &chessrl::SelfPlayMetrics::positions_generated)
        .def_readonly("nodes_evaluated", &chessrl::SelfPlayMetrics::nodes_evaluated)
        .def_readonly("runtime_seconds", &chessrl::SelfPlayMetrics::runtime_seconds);

    py::class_<chessrl::SelfPlayInferenceStats>(m, "SelfPlayInferenceStats")
        .def_readonly("batches", &chessrl::SelfPlayInferenceStats::batches)
        .def_readonly("positions_in_batches", &chessrl::SelfPlayInferenceStats::positions_in_batches)
        .def_readonly("total_inference_ms", &chessrl::SelfPlayInferenceStats::total_inference_ms)
        .def_readonly("encoder_calls", &chessrl::SelfPlayInferenceStats::encoder_calls)
        .def_readonly("total_encoder_ms", &chessrl::SelfPlayInferenceStats::total_encoder_ms);

    py::class_<chessrl::SelfPlayWorker>(m, "SelfPlayWorker")
        .def(py::init([](const std::string& engine_path,
                         const std::string& output_path,
                         int batch_size,
                         int threads,
                         int hash_size,
                         int max_games,
                         size_t max_buffer_size,
                         bool save_pgn,
                         std::string pgn_dir,
                         int simulations,
                         float cpuct,
                         float temperature,
                         float temperature_at_root,
                         bool use_dirichlet,
                         float dirichlet_alpha,
                         float dirichlet_epsilon,
                         int random_opening_plies) {
            chessrl::SelfPlayConfig config;
            config.engine_path = engine_path;
            config.output_path = output_path;
            config.concurrency = batch_size;
            config.threads = threads;
            config.hash_size = hash_size;
            config.max_games = max_games;
            config.max_buffer_size = max_buffer_size;
            config.save_pgn = save_pgn;
            config.pgn_dir = pgn_dir;
            config.simulations = simulations;
            config.cpuct = cpuct;
            config.temperature = temperature;
            config.temperature_at_root = temperature_at_root;
            config.use_dirichlet = use_dirichlet;
            config.dirichlet_alpha = dirichlet_alpha;
            config.dirichlet_epsilon = dirichlet_epsilon;
            config.random_opening_plies = random_opening_plies;
            return new chessrl::SelfPlayWorker(config);
        }),
        py::arg("engine_path"),
        py::arg("output_path"),
        py::arg("batch_size") = 8,
        py::arg("threads") = 1,
        py::arg("hash_size") = 4096,
        py::arg("max_games") = -1,
        py::arg("max_buffer_size") = 1000000,
        py::arg("save_pgn") = false,
        py::arg("pgn_dir") = "pgns",
        py::arg("simulations") = 800,
        py::arg("cpuct") = 0.65f,
        py::arg("temperature") = 1.3f,
        py::arg("temperature_at_root") = 1.0f,
        py::arg("use_dirichlet") = true,
        py::arg("dirichlet_alpha") = 0.12f,
        py::arg("dirichlet_epsilon") = 0.05f,
        py::arg("random_opening_plies") = 0)
        .def("step", &chessrl::SelfPlayWorker::step, py::call_guard<py::gil_scoped_release>())
        .def("inference_stats", &chessrl::SelfPlayWorker::inference_stats)
        .def("get_buffer_size", &chessrl::SelfPlayWorker::get_buffer_size);
}

