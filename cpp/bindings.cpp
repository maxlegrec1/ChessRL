#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>

#include "chess-library/include/chess.hpp"
#include "inference.hpp"
#include <cuda_runtime_api.h>

#include "mcts/common.hpp"
#include "mcts/single_inference.hpp"
#include "mcts/expand_kernel.hpp"
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

    // Packed-path parity hook: build the same (board, history), pack into the
    // ~776 B snapshot, then CPU-expand it. MUST be bit-identical to
    // encode_position_with_history (tests/test_packed_parity.py gates this).
    m.def(
        "encode_position_packed_roundtrip",
        [](const std::string& fen, const std::vector<std::string>& moves) {
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
                        "encode_position_packed_roundtrip: failed to parse move " + mv_uci);
                }
                board.makeMove(mv);
            }
            chessrl::mcts::HistoryStack stack =
                chessrl::mcts::build_root_history_stack(starting_fen, moves, board);
            auto packed = chessrl::mcts::single::pack_small_position(board, stack);
            return chessrl::mcts::single::expand_packed_to_dense(packed);
        },
        py::arg("fen"),
        py::arg("moves"),
        "Pack a FEN+history then CPU-expand it; must match encode_position_with_history.");

    // GPU-path parity hook: pack on host, upload, run the CUDA expansion
    // kernel (fp32 for an exact compare), download. Must equal the CPU
    // reference (tests/test_packed_gpu_parity.py gates this).
    m.def(
        "expand_packed_gpu",
        [](const std::string& fen, const std::vector<std::string>& moves) {
            const std::string starting_fen = fen.empty()
                ? std::string(chess::constants::STARTPOS)
                : fen;
            chess::Board board(starting_fen);
            for (const auto& mv_uci : moves)
            {
                chess::Move mv = chess::uci::uciToMove(board, mv_uci);
                if (mv == chess::Move::NO_MOVE)
                    throw std::runtime_error("expand_packed_gpu: bad move " + mv_uci);
                board.makeMove(mv);
            }
            chessrl::mcts::HistoryStack stack =
                chessrl::mcts::build_root_history_stack(starting_fen, moves, board);
            auto packed = chessrl::mcts::single::pack_small_position(board, stack);

            using chessrl::mcts::single::SmallPackedPosition;
            using chessrl::mcts::single::kPackElems;
            SmallPackedPosition* dPacked = nullptr;
            float* dOut = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&dPacked), sizeof(SmallPackedPosition)) != cudaSuccess
                || cudaMalloc(reinterpret_cast<void**>(&dOut), sizeof(float) * kPackElems) != cudaSuccess)
            {
                if (dPacked) cudaFree(dPacked);
                if (dOut) cudaFree(dOut);
                throw std::runtime_error("expand_packed_gpu: cudaMalloc failed");
            }
            cudaMemcpy(dPacked, &packed, sizeof(packed), cudaMemcpyHostToDevice);
            auto err = chessrl::mcts::single::launch_expand_planes(
                dPacked, dOut, 1, /*out_is_half=*/0, /*stream=*/0);
            cudaDeviceSynchronize();
            std::vector<float> host(static_cast<size_t>(kPackElems));
            if (err == cudaSuccess)
                cudaMemcpy(host.data(), dOut, sizeof(float) * kPackElems,
                           cudaMemcpyDeviceToHost);
            cudaFree(dPacked);
            cudaFree(dOut);
            if (err != cudaSuccess)
                throw std::runtime_error("expand_packed_gpu: kernel launch failed");
            return host;
        },
        py::arg("fen"),
        py::arg("moves"),
        "Pack + GPU-expand (fp32); must match encode_position_packed_roundtrip.");

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
           float dirichlet_epsilon,
           int virtual_loss_width) {
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
                options.virtual_loss_width = virtual_loss_width;
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
        py::arg("virtual_loss_width") = 1,
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
           float dirichlet_epsilon,
           int virtual_loss_width) {
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
                options.virtual_loss_width = virtual_loss_width;
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
        py::arg("virtual_loss_width") = 1,
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

    py::class_<chessrl::SelfPlayConfig>(m, "SelfPlayConfig")
        .def(py::init<>())
        .def_readwrite("engine_path", &chessrl::SelfPlayConfig::engine_path)
        .def_readwrite("output_path", &chessrl::SelfPlayConfig::output_path)
        .def_readwrite("concurrency", &chessrl::SelfPlayConfig::concurrency)
        .def_readwrite("threads", &chessrl::SelfPlayConfig::threads)
        .def_readwrite("hash_size", &chessrl::SelfPlayConfig::hash_size)
        .def_readwrite("simulations", &chessrl::SelfPlayConfig::simulations)
        .def_readwrite("cpuct", &chessrl::SelfPlayConfig::cpuct)
        .def_readwrite("temperature", &chessrl::SelfPlayConfig::temperature)
        .def_readwrite("temperature_at_root", &chessrl::SelfPlayConfig::temperature_at_root)
        .def_readwrite("virtual_loss_width", &chessrl::SelfPlayConfig::virtual_loss_width)
        .def_readwrite("use_dirichlet", &chessrl::SelfPlayConfig::use_dirichlet)
        .def_readwrite("dirichlet_alpha", &chessrl::SelfPlayConfig::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &chessrl::SelfPlayConfig::dirichlet_epsilon)
        .def_readwrite("max_games", &chessrl::SelfPlayConfig::max_games)
        .def_readwrite("max_buffer_size", &chessrl::SelfPlayConfig::max_buffer_size)
        .def_readwrite("save_pgn", &chessrl::SelfPlayConfig::save_pgn)
        .def_readwrite("pgn_dir", &chessrl::SelfPlayConfig::pgn_dir)
        .def_readwrite("random_opening_plies", &chessrl::SelfPlayConfig::random_opening_plies)
        .def_readwrite("temperature_plies", &chessrl::SelfPlayConfig::temperature_plies)
        .def_readwrite("temperature_late", &chessrl::SelfPlayConfig::temperature_late)
        .def_readwrite("resign_threshold", &chessrl::SelfPlayConfig::resign_threshold)
        .def_readwrite("resign_consecutive", &chessrl::SelfPlayConfig::resign_consecutive)
        .def_readwrite("resign_playthrough", &chessrl::SelfPlayConfig::resign_playthrough)
        .def_readwrite("max_plies", &chessrl::SelfPlayConfig::max_plies)
        .def_readwrite("pcr_full_prob", &chessrl::SelfPlayConfig::pcr_full_prob)
        .def_readwrite("pcr_fast_simulations", &chessrl::SelfPlayConfig::pcr_fast_simulations)
        .def_readwrite("quick_policy_temperature", &chessrl::SelfPlayConfig::quick_policy_temperature)
        .def_readwrite("fifo_buffer", &chessrl::SelfPlayConfig::fifo_buffer)
        .def_readwrite("curriculum_prob", &chessrl::SelfPlayConfig::curriculum_prob)
        .def_readwrite("curriculum_max_pieces", &chessrl::SelfPlayConfig::curriculum_max_pieces);

    py::class_<chessrl::SelfPlayStats>(m, "SelfPlayStats")
        .def_readonly("games", &chessrl::SelfPlayStats::games)
        .def_readonly("white_wins", &chessrl::SelfPlayStats::white_wins)
        .def_readonly("black_wins", &chessrl::SelfPlayStats::black_wins)
        .def_readonly("draws", &chessrl::SelfPlayStats::draws)
        .def_readonly("game_plies", &chessrl::SelfPlayStats::game_plies)
        .def_readonly("end_checkmate", &chessrl::SelfPlayStats::end_checkmate)
        .def_readonly("end_stalemate", &chessrl::SelfPlayStats::end_stalemate)
        .def_readonly("end_insufficient", &chessrl::SelfPlayStats::end_insufficient)
        .def_readonly("end_fifty", &chessrl::SelfPlayStats::end_fifty)
        .def_readonly("end_repetition", &chessrl::SelfPlayStats::end_repetition)
        .def_readonly("end_max_plies", &chessrl::SelfPlayStats::end_max_plies)
        .def_readonly("end_resign", &chessrl::SelfPlayStats::end_resign)
        .def_readonly("playthrough_games", &chessrl::SelfPlayStats::playthrough_games)
        .def_readonly("playthrough_would_resign", &chessrl::SelfPlayStats::playthrough_would_resign)
        .def_readonly("playthrough_false_resign", &chessrl::SelfPlayStats::playthrough_false_resign)
        .def_readonly("curriculum_games", &chessrl::SelfPlayStats::curriculum_games)
        .def_readonly("full_searches", &chessrl::SelfPlayStats::full_searches)
        .def_readonly("fast_searches", &chessrl::SelfPlayStats::fast_searches)
        .def_readonly("policy_moves", &chessrl::SelfPlayStats::policy_moves)
        .def_readonly("records_written", &chessrl::SelfPlayStats::records_written)
        .def_readonly("engine_swaps", &chessrl::SelfPlayStats::engine_swaps);

    py::class_<chessrl::SelfPlayWorker>(m, "SelfPlayWorker")
        .def(py::init<const chessrl::SelfPlayConfig&>(), py::arg("config"))
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
                         int random_opening_plies,
                         int virtual_loss_width) {
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
            config.virtual_loss_width = virtual_loss_width;
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
        py::arg("random_opening_plies") = 0,
        py::arg("virtual_loss_width") = 1)
        .def("step", &chessrl::SelfPlayWorker::step, py::call_guard<py::gil_scoped_release>())
        .def("inference_stats", &chessrl::SelfPlayWorker::inference_stats)
        .def("get_buffer_size", &chessrl::SelfPlayWorker::get_buffer_size)
        .def("set_engine", &chessrl::SelfPlayWorker::set_engine, py::arg("engine_path"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_config", &chessrl::SelfPlayWorker::set_config, py::arg("config"))
        .def("stats", &chessrl::SelfPlayWorker::stats);
}

