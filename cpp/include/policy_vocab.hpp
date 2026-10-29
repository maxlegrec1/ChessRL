#pragma once

// The policy vocabulary: the network's 1,929 move tokens, read once from
// cpp/policy_index.txt (relative to the working directory, so run from the repo root).
// Implemented in policy_vocab.cpp, which needs no TensorRT, so the search and the C++ tests
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
    static constexpr std::size_t kSize = 1929;

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
