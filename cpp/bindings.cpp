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

    py::class_<chessrl::MoveGenerationResult>(m, "MoveGenerationResult")
        .def_property_readonly("moves", [](const chessrl::MoveGenerationResult& self) { return self.moves; })
        .def_property_readonly("timings", [](const chessrl::MoveGenerationResult& self) { return self.timings; });

    m.def(
        "generate_moves",
        [](const std::vector<std::string>& fens, float temperature, int iterations, const std::string& engine_path) {
            return chessrl::generate_moves(fens, temperature, iterations, engine_path);
        },
        py::arg("fens"),
        py::arg("temperature"),
        py::arg("iterations") = 1,
        py::arg("engine_path") = std::string("model.trt"),
        "Run the TensorRT policy head and return sampled moves with timing stats.");
}

