Implementation Plan: Enforcing Dynamic Batching

1. Objective

Transition the inference engine from "Static/Padding-Safe" to "Strictly Dynamic".
Current State: The code detects if an engine is static (batch=1024) and pads inputs if necessary. This allowed the 32ms latency bug to persist silently.
Target State: The code must demand a dynamic engine and process exact batch sizes (e.g., 256) without any padding overhead.

2. Code Modifications

File: src/single_inference.cpp

Function: SmallBatchRunner::Impl::process

Step A: Enforce Dynamic Dimensions

Replace the "detection" logic with a "validation" logic.

Old: if (d[0] == -1) ... else ...

New:

if (inputDims.d[0] != -1) {
    throw std::runtime_error("Loaded engine is static! Please generate a dynamic engine with trtexec --minShapes/--optShapes/--maxShapes.");
}


Step B: Remove Padding Logic

Simplify the batch size calculation. We no longer need to calculate engineBatch distinct from actualCount.

Change: const size_t engineBatch = actualCount;

Step C: Simplify Copy Loops

Since engineBatch == actualCount, the loops no longer need to handle the "padding" case (where i >= actualCount).

Loop Cleanup:

Remove (i < actualCount) ? ... : ... ternary checks.

Simply access work[i] directly.

Function: SmallBatchRunner::Impl (Constructor)

Optimization (Optional): Ensure the flush_threshold_ is respected but allow partial batches to flow through dynamic execution immediately if the queue isn't filling fast enough (already handled by your Phase C wait logic).

3. Expected Outcome

Code Reduction: The copy loops become standard memcpy operations without conditional logic.

Performance: Latency for batch=250 will drop from ~32ms (processing 1024 items) to ~1-2ms (processing 250 items).

Safety: If a user accidentally generates a static engine, the program will crash immediately with a helpful error instead of running silently slow.