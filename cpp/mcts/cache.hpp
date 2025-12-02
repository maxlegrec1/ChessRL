#pragma once

#include "inference.hpp"
#include <atomic>
#include <mutex>
#include <optional>
#include <vector>
#include <cstdint>
#include <memory>
#include <iostream>
#include <fstream>

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
    explicit TranspositionTable(size_t size_mb)
    {
        // Calculate number of entries
        // Each entry is roughly 8 + sizeof(SmallPolicyValue) bytes
        // SmallPolicyValue has vector<float> policy_logits (1929 * 4 bytes) and array<float, 3> wdl (12 bytes)
        // Wait, vector is dynamic. We can't store it directly in a fixed array easily without pointers.
        // But we want a simple TT.
        // If we store full objects, we need to be careful about memory.
        // 1929 floats = ~7.7 KB.
        // If we want a large TT, we might need to store just the logits array, or use a shared pointer?
        // But SmallPolicyValue uses std::vector.
        // Let's use a simple vector of entries with a mutex for now, or a sharded map.
        // Given the size (7KB per entry), 128MB can store ~16,000 entries.
        // That's not a lot.
        // But maybe enough for a batch?
        // Actually, we can just use a large LRU cache or a simple hash table with replacement.
        
        // Let's use a simple vector of buckets.
        // To avoid storing 7KB in the entry, we can store a shared_ptr?
        // But we want to avoid allocations if possible.
        // However, SmallPolicyValue already has a vector.
        
        // Let's just use a simple implementation first.
        // We can optimize memory layout later.
        
        size_t entry_size = sizeof(CacheEntry) + 1929 * sizeof(float); // Approx
        num_entries_ = (size_mb * 1024 * 1024) / entry_size;
        if (num_entries_ < 1024) num_entries_ = 1024;
        
        entries_.resize(num_entries_);
        locks_.reserve(num_entries_ / 1024 + 1);
        for (size_t i = 0; i < num_entries_ / 1024 + 1; ++i)
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
        std::ofstream logfile("/tmp/profile.log", std::ios::app);
        logfile << "[Cache] Hits: " << hits_ << ", Misses: " << misses_ 
                << ", Ratio: " << (hits_ * 100.0 / (hits_ + misses_ + 1)) << "%" << std::endl;
    }

private:
    std::vector<CacheEntry> entries_;
    std::vector<std::unique_ptr<std::mutex>> locks_;
    size_t num_entries_;
    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};
};

} // namespace chessrl::mcts
