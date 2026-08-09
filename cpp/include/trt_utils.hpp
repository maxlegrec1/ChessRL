#pragma once

// Shared TensorRT helper utilities used by both the python-binding inference
// path (cpp/inference.cpp) and the in-process MCTS inference (cpp/mcts/single_inference.cpp).

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace chessrl::trt
{

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::clog << "[TensorRT] " << msg << std::endl;
        }
    }
};

inline size_t element_size(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        throw std::runtime_error("Unsupported TensorRT data type");
    }
}

inline size_t volume(const nvinfer1::Dims& dims)
{
    return std::accumulate(
        dims.d, dims.d + dims.nbDims,
        static_cast<size_t>(1),
        [](size_t a, int64_t b) { return a * static_cast<size_t>(b); });
}

inline std::string resolve_engine_path(const std::string& engine_path, const std::string& default_name)
{
    std::string resolved = engine_path.empty() ? default_name : engine_path;
    try
    {
        resolved = std::filesystem::absolute(resolved).string();
    }
    catch (...)
    {
    }
    return resolved;
}

} // namespace chessrl::trt
