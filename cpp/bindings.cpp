#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "inference.hpp"

namespace py = pybind11;

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

    py::class_<chessrl::LeelaInferenceResult>(m, "LeelaInferenceResult")
        .def_property_readonly("moves", [](const chessrl::LeelaInferenceResult& self) { return self.moves; })
        .def_property_readonly(
            "value_winner", [](const chessrl::LeelaInferenceResult& self) { return self.value_winner; })
        .def_property_readonly("value_q", [](const chessrl::LeelaInferenceResult& self) { return self.value_q; })
        .def_property_readonly("timings", [](const chessrl::LeelaInferenceResult& self) { return self.timings; });

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
        "leela_generate_from_move_lists",
        [](const std::vector<std::string>& origin_fens,
           const std::vector<std::vector<std::string>>& move_lists,
           float temperature,
           int iterations,
           const std::string& engine_path) {
            return chessrl::leela_generate_from_move_lists(origin_fens, move_lists, temperature, iterations, engine_path);
        },
        py::arg("origin_fens"),
        py::arg("move_lists"),
        py::arg("temperature"),
        py::arg("iterations") = 1,
        py::arg("engine_path") = std::string("leela.trt"),
        "Run Leela TensorRT inference on batched move histories and return selected moves and value heads.");
}

