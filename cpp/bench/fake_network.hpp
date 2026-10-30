#pragma once

// A stand-in for the CUDA network, to run the real search on a CPU
// (scripts/build_cpu_bench.sh). fake_network.cpp implements SmallBatchRunner's
// interface and is linked instead of single_inference.cpp, so the search code is
// the production code, unchanged.
//
// Each position gets fixed pseudo-random outputs from a hash of its current board:
// policy logits drawn from N(0, sharpness^2) (a trained network's priors are sharp,
// which makes deeper trees), a WDL triple and a moves-left estimate. With
// delay_us = 0 every request is answered at once in the calling thread, so a run
// measures the search's own CPU cost. With delay_us > 0, requests are batched like
// the real runner (dispatched at max_batch or flush_us after the first arrives) and
// answered delay_us later, as a simulated GPU.

#include <cstdint>
#include <vector>

#include "mcts/packed_position.hpp"

namespace chessrl::bench
{

struct FakeNetworkConfig
{
    float sharpness{2.0F};
    std::uint64_t seed{1};
    int delay_us{0};
    int flush_us{100};
    int max_batch{128};
};

// Call once, before the first search.
void configure_fake_network(const FakeNetworkConfig& config);

struct FakeNetworkStats
{
    std::uint64_t evaluations{0};
    std::uint64_t batches{0};
};
FakeNetworkStats fake_network_stats();

// Nanoseconds the fake network spends per evaluation on these positions (its own
// cost, to subtract from the search's CPU time in immediate mode).
double fake_network_cost_ns(const std::vector<mcts::single::SmallPackedPosition>& positions, int reps);

} // namespace chessrl::bench
