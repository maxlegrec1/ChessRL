#pragma once

#include "inference.hpp"
#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace chessrl::mcts
{

struct CacheEntry
{
    uint64_t key;
    SmallPolicyValue value;
};

class TranspositionTable
{
public:
    // Each entry stores a SmallPolicyValue (1929 float logits + 3 WDL floats); entries are
    // sharded into 1024-bucket lock groups for low-contention concurrent access.
    explicit TranspositionTable(size_t size_mb)
    {
        const size_t entry_size = sizeof(CacheEntry) + 1929 * sizeof(float);
        num_entries_ = std::max<size_t>(1024, (size_mb * 1024 * 1024) / entry_size);

        entries_.resize(num_entries_);
        const size_t num_locks = num_entries_ / 1024 + 1;
        locks_.reserve(num_locks);
        for (size_t i = 0; i < num_locks; ++i)
        {
            locks_.push_back(std::make_unique<std::mutex>());
        }
    }

    std::optional<SmallPolicyValue> get(uint64_t key)
    {
        size_t idx = key % num_entries_;
        size_t lock_idx = idx / 1024;
        
        std::lock_guard<std::mutex> lock(*locks_[lock_idx]);
        if (entries_[idx].key == key)
        {
            // Verify value is valid (not empty)
            if (!entries_[idx].value.policy_logits.empty())
            {
                hits_++;
                return entries_[idx].value;
            }
        }
        misses_++;
        return std::nullopt;
    }

    void put(uint64_t key, const SmallPolicyValue& value)
    {
        size_t idx = key % num_entries_;
        size_t lock_idx = idx / 1024;
        
        std::lock_guard<std::mutex> lock(*locks_[lock_idx]);
        entries_[idx].key = key;
        entries_[idx].value = value;
    }

    void print_stats() const
    {
        const size_t h = hits_.load();
        const size_t m = misses_.load();
        std::cerr << "[Cache] hits=" << h << " misses=" << m
                  << " ratio=" << (h * 100.0 / (h + m + 1)) << "%\n";
    }

    size_t hits() const { return hits_.load(); }
    size_t misses() const { return misses_.load(); }

private:
    std::vector<CacheEntry> entries_;
    std::vector<std::unique_ptr<std::mutex>> locks_;
    size_t num_entries_;
    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};
};

} // namespace chessrl::mcts
