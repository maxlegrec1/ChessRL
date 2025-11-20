from src.mcts import SmallMCTS
import time
import matplotlib.pyplot as plt
import numpy as np

# Initialize MCTS
small = SmallMCTS(engine_path="model_dynamic.trt", temperature=1, cpuct=1.0)
fen = "r2qkb1r/1p2pppp/p1n1b3/2nN4/2B5/5N2/PPP2PPP/R1BQK2R w KQkq - 0 9"

# Batch sizes to test
batch_sizes = [256, 512, 1024]
num_runs = 2

# Store results: batch_size -> list of times per position
results = {}

print("Benchmarking batch sizes...")
for batch_size in batch_sizes:
    print(f"\nTesting batch size: {batch_size}")
    times_per_pos = []
    
    for run in range(num_runs):
        start_time = time.time()
        res = small.run([fen] * batch_size, temperature=1, simulations=600, cpuct=1.0)
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_position = total_time / batch_size
        times_per_pos.append(time_per_position)
        
        print(f"  Run {run + 1}/{num_runs}: {time_per_position:.4f} s/pos")
    
    avg_time_per_pos = np.mean(times_per_pos)
    results[batch_size] = avg_time_per_pos
    print(f"  Average: {avg_time_per_pos:.4f} s/pos")

# Plot results
print("\nPlotting results...")
batch_sizes_list = list(results.keys())
avg_times = list(results.values())

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes_list, avg_times, marker='o', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Average Time per Position (seconds)', fontsize=12)
plt.title('MCTS Performance: Average Time per Position vs Batch Size', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xscale('log', base=2)
plt.xticks(batch_sizes_list, batch_sizes_list)
plt.tight_layout()

# Save the plot
plt.savefig('batch_size_benchmark.png', dpi=150)
print("Plot saved as 'batch_size_benchmark.png'")

# Display the plot
plt.show()

# Print summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for batch_size in batch_sizes_list:
    print(f"Batch size {batch_size:4d}: {results[batch_size]:.4f} s/pos")

