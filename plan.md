Refactoring Plan: High-Performance MCTS & Crash Fix

1. Objective

The current implementation of the Parallel MCTS search has two critical failures:

Stability: Spawning one thread per game (e.g., 512 threads for 512 FENs) causes stack exhaustion and system crashes.

Performance: The TreeNode object is too heavy (stores full chess::Board), and mutex contention in the inference queue is high.

Goal: Refactor the codebase to use a fixed-size Thread Pool for game simulations and optimize memory bandwidth by removing the Board state from the search tree.

2. Macro Plan

This refactor must be applied in three distinct phases to ensure correctness.

Phase A: Architecture Fix (Stop the Crash)

Modify run_small_mcts_batch in mcts_small.cpp.

Replace "Thread-Per-Game" with a "Worker Pool" pattern using std::thread::hardware_concurrency().

Phase B: Memory Optimization (Speed Up)

Modify struct TreeNode in mcts_small.cpp.

Remove chess::Board board from the node. Store only chess::Move.

Implement state reconstruction during tree traversal.

Phase C: Inference Tuning (Reduce Latency)

Modify single_inference.cpp.

Remove artificial sleep/timeouts to maximize GPU throughput.

3. Implementation Details

Phase A: Implement Worker Pool

File: mcts_small.cpp
Function: run_small_mcts_batch

Current State:
The code loops through batch_options and calls workers.emplace_back for every item.

Required Changes:

Determine the number of threads: size_t num_threads = std::max(1u, std::thread::hardware_concurrency());

Create an atomic counter: std::atomic<size_t> next_job_index{0};

Create a worker_routine lambda that:

Loops while (true).

Fetches size_t job_id = next_job_index.fetch_add(1);.

Breaks if job_id >= batch_options.size().

Instantiates SmallMctsRunner for batch_options[job_id].

Runs the simulation and stores the result in summaries[job_id].

Spawn only num_threads into the workers vector.

Join threads.

Phase B: Slim TreeNode & Dynamic State

File: mcts_small.cpp

1. Modify TreeNode Struct

Remove the heavy chess::Board object.

struct TreeNode
{
    // REMOVED: chess::Board board;
    
    // KEPT:
    int visit_count{0};
    float value_sum{0.0F};
    float value_prior{0.0F};
    std::array<float, 3> wdl{};
    int parent{-1};
    
    // ADDED:
    chess::Move move_from_parent; // The move that led to this node
    
    // KEPT:
    std::vector<int> children;
    bool expanded{false};
    bool terminal{false};
};


2. Update SmallMctsRunner

Since node.board no longer exists, the board state must be recalculated as we traverse down the tree.

Modify run_simulation():

Maintain a local chess::Board current_board variable, initialized to the root board at the start of the simulation.

Inside the traversal loop, when selecting a child:

// Apply the move to our local board tracker
current_board.makeMove(tree_.nodes[child_index].move_from_parent);


Modify evaluate_node(int node_index):

Signature change: evaluate_node(int node_index, const chess::Board& board_state)

Use the passed board_state for checking terminal_value and for encoding features for the neural network.

Modify add_child logic:

When expanding, store the move in the new TreeNode. Do not store the resulting board.

Phase C: Inference Engine Tuning

File: single_inference.cpp

Disable Forced Latency:
In SmallBatchRunner::Impl, find flush_timeout_.

Change: Set it to 0 or std::chrono::microseconds(1).

Reason: With the worker pool active, the GPU queue will fill up naturally. We don't want the worker threads sleeping waiting for a timeout; we want them pushing data.

Optimize runSmallInference:
Ensure that cudaStreamSynchronize is used efficiently. (The current implementation is acceptable for Phase 1, but ensure process() doesn't hold the lock longer than necessary).

4. Validation Checklist (For the AI)

Compile Check: Ensure chess::Board is not referenced as a member of TreeNode.

Logic Check: Verify that current_board in run_simulation is reset to root at the start of every rollout.

Crash Test: Run with [fen] * 500. Monitor RAM usage. It should remain stable (stack size constant).

Output Check: Verify the JSON output format remains identical to the original implementation.