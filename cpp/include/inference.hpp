#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace chessrl
{

struct SmallPolicyValue
{
    std::vector<float> policy_logits;
    std::array<float, 3> wdl;
    // Both MLH heads are linear (no activation in the engine). The raw output
    // is interpretable as plies_remaining / 25 — multiply by 25 for raw
    // plies, apply tanh for a bounded tiebreaker modifier.
    //   mlh   ← true target head (game-end plies).
    //   mlh_q ← MCTS-backed target head; used by MCTS for depth-incremented
    //           backup AND for the selection tiebreaker.
    float mlh{0.0F};
    float mlh_q{0.0F};
};

size_t small_policy_size();

const std::vector<std::string>& small_policy_vocabulary();

} // namespace chessrl


