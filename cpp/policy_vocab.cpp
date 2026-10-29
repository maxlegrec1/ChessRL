#include "include/policy_vocab.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference.hpp"

namespace chessrl
{
namespace vocab
{

PolicyVocab::PolicyVocab()
{
    std::ifstream file(kVocabPath);
    if (!file)
    {
        throw std::runtime_error(std::string("Failed to open policy vocab file: ") + kVocabPath);
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        tokens.emplace_back(line);
        if (!line.empty())
        {
            toIndex[line] = static_cast<int>(tokens.size() - 1);
        }
    }

    if (tokens.size() != kSize)
    {
        throw std::runtime_error("Unexpected small policy vocab size: " + std::to_string(tokens.size()));
    }
}

std::optional<int> PolicyVocab::indexOf(const std::string& token) const
{
    auto it = toIndex.find(token);
    if (it == toIndex.end())
    {
        return std::nullopt;
    }
    return it->second;
}

PolicyVocab& getPolicyVocab()
{
    static PolicyVocab vocab;
    return vocab;
}

} // namespace vocab

size_t small_policy_size()
{
    return vocab::PolicyVocab::kSize;
}

const std::vector<std::string>& small_policy_vocabulary()
{
    return vocab::getPolicyVocab().tokensRef();
}

} // namespace chessrl
