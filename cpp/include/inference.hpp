#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace chessrl
{

inline float half_to_float(std::uint16_t h)
{
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    const std::uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    if (exp == 0)
    {
        const float f = static_cast<float>(mant) * 5.9604645e-8f;  // subnormal: mant * 2^-24
        return sign ? -f : f;
    }
    const std::uint32_t bits = sign | ((exp == 31 ? 255u : exp + 112u) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

struct SmallPolicyValue
{
    std::vector<float> policy_logits;
    // A result taken from the fused network's output buffer leaves its logits there, in FP16, and
    // the search reads only the legal moves' entries through logit(); valid while the result's
    // ticket is held (policy_logits is then empty).
    const std::uint16_t* policy_fp16{nullptr};
    int policy_fp16_size{0};
    std::array<float, 3> wdl;
    // Both MLH heads are linear (no activation in the engine). The raw output
    // is interpretable as plies_remaining / 25 — multiply by 25 for raw
    // plies, apply tanh for a bounded tiebreaker modifier.
    //   mlh   ← true target head (game-end plies).
    //   mlh_q ← MCTS-backed target head; used by MCTS for depth-incremented
    //           backup AND for the selection tiebreaker.
    float mlh{0.0F};
    float mlh_q{0.0F};

    size_t policy_size() const
    {
        return policy_fp16 ? static_cast<size_t>(policy_fp16_size) : policy_logits.size();
    }
    float logit(size_t i) const
    {
        return policy_fp16 ? half_to_float(policy_fp16[i]) : policy_logits[i];
    }
    // A copy that owns its logits, for what outlives the ticket (the transposition table).
    SmallPolicyValue owned() const
    {
        SmallPolicyValue v = *this;
        if (policy_fp16)
        {
            v.policy_logits.resize(static_cast<size_t>(policy_fp16_size));
            for (size_t i = 0; i < v.policy_logits.size(); ++i) v.policy_logits[i] = half_to_float(policy_fp16[i]);
            v.policy_fp16 = nullptr;
            v.policy_fp16_size = 0;
        }
        return v;
    }
};

size_t small_policy_size();

const std::vector<std::string>& small_policy_vocabulary();

} // namespace chessrl


