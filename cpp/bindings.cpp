#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "inference.hpp"
#include "mcts/small_runner.hpp"
#include "mcts/summary.hpp"

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
}

