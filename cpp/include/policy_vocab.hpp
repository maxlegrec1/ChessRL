#pragma once

// The policy vocabulary: the side to move's moves in white's view (positions with black to move are
// mirrored), the network's 1,858 move tokens (lc0's set), read once from
// cpp/policy_index.txt (relative to the working directory, so run from the repo root).
// Implemented in policy_vocab.cpp, which needs no CUDA, so the search and the C++ tests
// build without it.

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace chessrl::vocab
{

class PolicyVocab
{
public:
    static constexpr std::size_t kSize = 1858;

    PolicyVocab();

    const std::string& at(int idx) const { return tokens.at(static_cast<size_t>(idx)); }

    std::optional<int> indexOf(const std::string& token) const;

    const std::vector<std::string>& tokensRef() const { return tokens; }

private:
    static constexpr const char* kVocabPath = "cpp/policy_index.txt";
    std::vector<std::string> tokens;
    std::unordered_map<std::string, int> toIndex;
};

PolicyVocab& getPolicyVocab();

} // namespace chessrl::vocab
