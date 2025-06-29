import os
import ray
import time
import random
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm
from stockfish import Stockfish
from typing import Dict, List, Any, Optional
import threading
from dataclasses import dataclass
from collections import defaultdict, deque
import psutil
import json

# Assuming utils.parse.dir_iterator exists as in the original code
from utils.parse_improved import batch_generator

# --- Performance Monitoring ---
@dataclass
class PerformanceMetrics:
    """Class to track performance metrics"""
    total_fens_processed: int = 0
    total_variations_generated: int = 0
    total_batches_processed: int = 0
    total_errors: int = 0
    worker_utilization: Dict[int, float] = None
    throughput_per_second: float = 0.0
    avg_processing_time_per_fen: float = 0.0
    start_time: float = 0.0
    
    def __post_init__(self):
        if self.worker_utilization is None:
            self.worker_utilization = {}

class PerformanceMonitor:
    """Real-time performance monitoring with thread-safe operations"""
    
    def __init__(self, num_workers: int):
        self.metrics = PerformanceMetrics()
        self.num_workers = num_workers
        self.lock = threading.Lock()
        self.recent_processing_times = deque(maxlen=100)  # Keep last 100 processing times
        self.worker_task_counts = defaultdict(int)
        self.worker_active_time = defaultdict(float)
        
    def record_batch_completion(self, batch_size: int, processing_time: float, errors: int):
        """Record completion of a batch"""
        with self.lock:
            self.metrics.total_fens_processed += batch_size
            self.metrics.total_batches_processed += 1
            self.metrics.total_errors += errors
            self.recent_processing_times.append(processing_time)
            
            # Update throughput calculation
            if self.recent_processing_times:
                avg_time = sum(self.recent_processing_times) / len(self.recent_processing_times)
                self.metrics.avg_processing_time_per_fen = avg_time / batch_size if batch_size > 0 else 0
                self.metrics.throughput_per_second = batch_size / avg_time if avg_time > 0 else 0
    
    def record_worker_activity(self, worker_id: int, active_time: float):
        """Record worker activity time"""
        with self.lock:
            self.worker_task_counts[worker_id] += 1
            self.worker_active_time[worker_id] += active_time
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self.lock:
            current_time = time.time()
            total_time = current_time - self.metrics.start_time
            
            # Calculate worker utilization
            worker_utilization = {}
            for worker_id in range(self.num_workers):
                if worker_id in self.worker_active_time:
                    utilization = (self.worker_active_time[worker_id] / total_time) * 100
                    worker_utilization[worker_id] = min(100.0, utilization)
                else:
                    worker_utilization[worker_id] = 0.0
            
            return {
                "total_fens_processed": self.metrics.total_fens_processed,
                "total_batches_processed": self.metrics.total_batches_processed,
                "total_errors": self.metrics.total_errors,
                "throughput_per_second": self.metrics.throughput_per_second,
                "avg_processing_time_per_fen": self.metrics.avg_processing_time_per_fen,
                "worker_utilization": worker_utilization,
                "avg_worker_utilization": sum(worker_utilization.values()) / len(worker_utilization) if worker_utilization else 0,
                "total_runtime": total_time,
                "system_cpu_percent": psutil.cpu_percent(),
                "system_memory_percent": psutil.virtual_memory().percent
            }

# --- Logging Configuration ---
def setup_logging(output_dir: str) -> logging.Logger:
    """Set up structured logging"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("chess_cot_generation")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = os.path.join(log_dir, f"chess_cot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# --- Configuration ---
DIR_PATH = "/media/maxime/385e67e0-8703-4723-a0bc-af3a292fd030/stockfish_data/Leela_training" # Or your actual path
STOCKFISH_BINARY_PATH = "stockfish/stockfish-ubuntu-x86-64-avx2" # Or your actual path - MAKE SURE THIS IS CORRECT
OUTPUT_DIR = "cot" # Directory to save results
BATCH_SIZE = 32   # Increased batch size for better throughput
NUM_WORKERS = os.cpu_count() or 4 # Use number of CPU cores, default 4 if undetectable
STOCKFISH_DEPTH = 15 # Minimum depth
# Threads per stockfish instance: Use 1 when running many parallel workers
STOCKFISH_THREADS_PER_INSTANCE = 1
# Parameters for variation generation
NUM_VARIATIONS_PER_FEN = 5
MIN_LEN_VARIATION = 3
MAX_LEN_VARIATION = 8
TOP_K_MOVES_TO_CONSIDER = 5  # Reduced from 20 to 5 since we only keep first 5 moves
# Parameter for weighted sampling - higher value means more focus on top moves
# Lower value (~1) means more uniform distribution
SOFTMAX_TEMPERATURE = 1.0 # Adjust this to control the 'peakiness' of the distribution

# Performance settings
PARQUET_COMPRESSION = 'snappy'  # Fast compression for parquet files
STATS_REPORTING_INTERVAL = 10  # Report stats every N batches
BATCH_WRITE_BUFFER_SIZE = 3  # Reduced buffer size but larger write chunks
WORKER_HEALTH_CHECK_INTERVAL = 50  # Check worker health every N batches

# --- Helper Function for Score Conversion ---
# Define a large score for mate situations
MATE_SCORE = 3000 # Arbitrarily large centipawn value for mate

def score_to_numeric(score_dict, player_pov=True):
    """
    Converts Stockfish score {'Centipawn': cp, 'Mate': None} or {'Centipawn': None, 'Mate': m}
    into a single numeric value for comparison/probability calculation.
    Uses tanh compression to keep scores between -5 and +5, allowing both mates and 
    very good moves to be considered together.
    player_pov=True means positive scores are good for the current player.
    """
    if score_dict.get("Mate") is not None:
        mate_in = score_dict["Mate"]
        if mate_in > 0: # Mate found for the current player
            # For positive mate, use a very high raw score
            # Faster mates get higher scores: mate in 1 = 1000, mate in 2 = 999, etc.
            raw_score = 1000 - mate_in
        elif mate_in < 0: # Mate found for the opponent
            # For negative mate, use a very low raw score
            # Faster opponent mates get lower scores: mate in -1 = -1000, mate in -2 = -999, etc.
            raw_score = -1000 - mate_in
        else: # Mate is 0? Should not happen, treat as 0 cp
             raw_score = 0
    elif score_dict.get("Centipawn") is not None:
        raw_score = score_dict["Centipawn"]
    else:
        # Should not happen with valid get_top_moves output
        raw_score = 0
    
    # Apply tanh compression: 5 * tanh(raw_score / 500)
    # This maps raw_score to approximately [-5, +5] range
    # Mate scores (~1000) -> ~5, very good moves (~500) -> ~4.6, decent moves (~200) -> ~1.9
    import math
    compressed_score = 5 * math.tanh(raw_score / 500)
    
    return compressed_score

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Ray Actor Definition ---
@ray.remote
class StockfishWorker:
    def __init__(self, stockfish_path, depth, threads, worker_id):
        self.worker_id = worker_id
        try:
            # Each worker gets its own Stockfish instance
            self.stockfish = Stockfish(path=stockfish_path, depth=depth)
            # Set parameters like Threads for this specific instance
            self.stockfish.update_engine_parameters({
                # MultiPV > 1 needed for get_top_moves to return multiple moves
                # Set it high enough to cover TOP_K_MOVES_TO_CONSIDER
                "MultiPV": 1,
                "Threads": threads,
                "Hash": 64  # Reduced hash size for better memory efficiency with multiple workers
            })
            print(f"Stockfish worker {worker_id} initialized (PID: {os.getpid()}) with depth {depth}, threads {threads}, MultiPV {max(1, TOP_K_MOVES_TO_CONSIDER)}")
        except Exception as e:
            print(f"Error initializing Stockfish in worker {worker_id}: {e}")
            self.stockfish = None # Mark as invalid

    def _make_stockfish_play(self, k, initial_fen):
        """
        Generates k moves starting from the current position using the BEST move.
        Resets to the initial_fen afterwards.
        (Helper method internal to the actor)
        """
        if not self.stockfish: return ["error: no stockfish"]

        moves = []
        try:
            for _ in range(k):
                # Use depth set during initialization
                move = self.stockfish.get_best_move()
                if move is None: # No legal moves (checkmate/stalemate)
                    move = "end"
                    moves.append(move)
                    break
                moves.append(move)
                # Make the move on the board for the next iteration
                try : 
                    self.stockfish.make_moves_from_current_position([move])
                except:
                    break

        except Exception as e:
             moves.append(f"error: {e}")
        finally:
             # VERY IMPORTANT: Reset position back to the original FEN for the next variation
             self.stockfish.set_fen_position(initial_fen)

        return moves

    def generate_variations_for_fen(self, fen, num_var=5, min_len_var=3, max_len_var=8, top_k=5, temperature=1.0):
        """
        Generates variations for a single FEN using this worker's Stockfish instance.
        Selects the top k moves where k is determined by:
        - Minimum 1, maximum 5 moves
        - Only moves within 100 centipawns (delta eval) of the best move
        """
        start_time = time.time()
        
        if not self.stockfish:
             return {"fen": fen, "variations": [], "error": "Stockfish not initialized", "processing_time": time.time() - start_time, "worker_id": self.worker_id}

        # Validate FEN first
        if not self.stockfish.is_fen_valid(fen):
            return {"fen": fen, "variations": [], "error": "Invalid FEN", "processing_time": time.time() - start_time, "worker_id": self.worker_id}

        variations = []
        try:
            self.stockfish.set_fen_position(fen)

            # Get top moves - This uses the instance's depth setting and MultiPV setting
            best_first_moves_data = self.stockfish.get_top_moves(top_k)

            if not best_first_moves_data: # No legal moves from this position
                 return {"fen": fen, "variations": [], "error": "No legal moves", "processing_time": time.time() - start_time, "worker_id": self.worker_id}

            # --- Dynamic k Selection Based on Evaluation Delta ---
            # Get the best move's score as baseline
            best_move_score = score_to_numeric(best_first_moves_data[0])
            
            # Find all moves within the threshold of the best move
            # Since scores are now compressed to [-5, +5] range, use 1.0 as threshold
            selected_moves = []
            for move_data in best_first_moves_data:
                move_score = score_to_numeric(move_data)
                # Calculate delta (best_score - current_score)
                # For the best move, delta = 0
                # For worse moves, delta will be positive (since they have lower scores)
                delta = best_move_score - move_score
                
                # Include moves with delta <= 1.0 (equivalent to ~100cp in compressed space)
                if delta <= 1.0:
                    selected_moves.append(move_data["Move"])
                else:
                    break  # Since moves are sorted by strength, we can break early
            
            # Apply minimum and maximum constraints
            k = max(1, min(5, len(selected_moves)))
            selected_moves = selected_moves[:k]
            
            # Generate lengths for each variation
            var_lengths = np.random.randint(min_len_var - 1, max_len_var - 1 + 1, size=k)

            results = []
            for i, (first_move, var_len) in enumerate(zip(selected_moves, var_lengths)):
                current_variation = []
                current_variation.append(first_move)

                # Check if the first move is valid before proceeding (robustness)
                self.stockfish.set_fen_position(fen) # Reset to original state for this variation
                self.stockfish.make_moves_from_current_position([first_move])

                # Generate the rest of the variation using the *best* continuation moves
                if var_len > 0:
                    # Make the first move again WITHIN the context of this loop iteration
                    self.stockfish.set_fen_position(fen) # Reset
                    self.stockfish.make_moves_from_current_position([first_move]) # Make the specific first move
                    fen_after_first = self.stockfish.get_fen_position() # FEN for continuation

                    # Now generate continuation from *this* state, but reset to original 'fen' at the end
                    continuation_moves = self._make_stockfish_play(var_len, fen) # Pass original FEN for final reset
                    current_variation.extend(continuation_moves)

                results.append(current_variation)

                # Reset position for the *next* iteration
                self.stockfish.set_fen_position(fen)

            processing_time = time.time() - start_time
            return {"fen": fen, "variations": results, "error": None, "processing_time": processing_time, "worker_id": self.worker_id}

        except Exception as e:
            import traceback
            processing_time = time.time() - start_time
            # Attempt to reset state in case of error
            try:
                self.stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") # Reset to start
            except:
                pass # Ignore error during reset attempt
            return {"fen": fen, "variations": [], "error": str(e), "processing_time": processing_time, "worker_id": self.worker_id}


# --- Optimized Data Management ---
class BatchDataManager:
    """Handles efficient writing of results to parquet files"""
    
    def __init__(self, output_dir: str, buffer_size: int = 5):
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.buffer = []
        self.batch_counter = 0
        self.file_counter = 0
        
    def add_batch_results(self, batch_results: List[Dict[str, Any]]):
        """Add batch results to buffer"""
        # Convert to flat records for parquet
        records = []
        for result in batch_results:
            if result.get("error") is None and result.get("variations"):
                for i, variation in enumerate(result["variations"]):
                    records.append({
                        "fen": result["fen"],
                        "variation_id": i,
                        "variation": " ".join(variation) if isinstance(variation, list) else str(variation),
                        "variation_length": len(variation) if isinstance(variation, list) else 1,
                        "processing_time": result.get("processing_time", 0),
                        "worker_id": result.get("worker_id", -1),
                        "batch_id": self.batch_counter
                    })
        
        self.buffer.extend(records)
        self.batch_counter += 1
        
        # Write if buffer is full (write larger chunks for better I/O performance)
        if len(self.buffer) >= self.buffer_size * BATCH_SIZE * 2:  # Write when we have 2x the buffer size
            self._write_buffer()
    
    def _write_buffer(self):
        """Write buffer to parquet file"""
        if not self.buffer:
            return
            
        df = pd.DataFrame(self.buffer)
        filename = f"chess_variations_{self.file_counter:06d}.parquet"
        filepath = os.path.join(self.output_dir, filename)
        
        # Use more efficient parquet writing options
        df.to_parquet(
            filepath, 
            compression=PARQUET_COMPRESSION, 
            index=False,
            engine='pyarrow',
            row_group_size=10000  # Optimize for better compression and query performance
        )
        
        print(f"Wrote {len(self.buffer)} records to {filename}")
        self.buffer.clear()
        self.file_counter += 1
    
    def flush(self):
        """Flush remaining buffer to file"""
        if self.buffer:
            self._write_buffer()

class WorkerHealthMonitor:
    """Monitor worker health and replace failed workers"""
    
    def __init__(self, workers, stockfish_path, depth, threads):
        self.workers = workers
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.threads = threads
        self.failed_workers = set()
        
    def check_and_replace_workers(self):
        """Check worker health and replace failed ones"""
        # Simple health check - try to get worker status
        health_checks = []
        for i, worker in enumerate(self.workers):
            if i not in self.failed_workers:
                try:
                    # Send a simple task to check if worker is responsive
                    health_checks.append(ray.get(worker.generate_variations_for_fen.remote(
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
                        num_var=1, min_len_var=1, max_len_var=1, top_k=1, temperature=1.0
                    ), timeout=5))  # 5 second timeout
                except Exception:
                    # Worker failed, mark for replacement
                    self.failed_workers.add(i)
                    self.workers[i] = StockfishWorker.remote(
                        self.stockfish_path, 
                        self.depth, 
                        self.threads, 
                        i
                    )
                    print(f"Replaced failed worker {i}")
        
        return len(self.failed_workers)

# --- Main Execution ---
def main():
    """Main processing function that runs the full parallel chess variation generation pipeline."""
    
    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("Starting chess COT generation with performance optimizations")
    logger.info(f"Configuration: Workers={NUM_WORKERS}, Batch_Size={BATCH_SIZE}, Top_K={TOP_K_MOVES_TO_CONSIDER}")
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(NUM_WORKERS)
    monitor.metrics.start_time = time.time()
    
    # Initialize Ray
    try:
        ray.init(num_cpus=NUM_WORKERS)
        logger.info(f"Ray initialized with {NUM_WORKERS} workers")
    except Exception as e:
        logger.warning(f"Ray initialization failed: {e}. Trying ignore_reinit_error=True")
        ray.init(num_cpus=NUM_WORKERS, ignore_reinit_error=True)

    # Create Stockfish worker actors
    logger.info("Creating Stockfish workers...")
    workers = [StockfishWorker.remote(STOCKFISH_BINARY_PATH, STOCKFISH_DEPTH, STOCKFISH_THREADS_PER_INSTANCE, i)
               for i in range(NUM_WORKERS)]
    # Small delay to allow workers to initialize
    time.sleep(2)
    logger.info(f"{len(workers)} Stockfish workers created")

    # Initialize the generator
    logger.info(f"Initializing FEN generator from: {DIR_PATH}")
    try:
        from utils.parse_improved import batch_generator
        fen_generator = batch_generator(DIR_PATH, return_fen=True, batch_size=BATCH_SIZE)
        logger.info("Using real dir_iterator")
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Could not import or find path for real dir_iterator: {e}. Using placeholder.")
        fen_generator = batch_generator(DIR_PATH, return_fen=True, batch_size=BATCH_SIZE)

    # Initialize data manager
    data_manager = BatchDataManager(OUTPUT_DIR, BATCH_WRITE_BUFFER_SIZE)
    
    # Initialize worker health monitor
    health_monitor = WorkerHealthMonitor(
        workers, 
        STOCKFISH_BINARY_PATH, 
        STOCKFISH_DEPTH, 
        STOCKFISH_THREADS_PER_INSTANCE
    )

    # Main processing loop
    total_batches_to_process = 1000000 
    logger.info(f"Starting processing loop for up to {total_batches_to_process} batches")
    
    for i in tqdm(range(total_batches_to_process), desc="Processing Batches"):
        batch_start_time = time.time()
        
        # Periodic worker health check
        if i > 0 and i % WORKER_HEALTH_CHECK_INTERVAL == 0:
            failed_count = health_monitor.check_and_replace_workers()
            if failed_count > 0:
                logger.warning(f"Replaced {failed_count} failed workers at batch {i}")
        
        try:
            fens = next(fen_generator)
        except StopIteration:
            logger.info("FEN generator finished")
            break
        except Exception as e:
            logger.error(f"Error getting next batch from generator: {e}")
            break

        if not fens:
            logger.warning("Received empty batch, stopping")
            break

        # Distribute tasks to workers with better load balancing
        tasks = []
        for idx, fen in enumerate(fens):
            # Skip failed workers in round-robin assignment
            worker_idx = idx % NUM_WORKERS
            while worker_idx in health_monitor.failed_workers:
                worker_idx = (worker_idx + 1) % NUM_WORKERS
            
            worker = workers[worker_idx]
            tasks.append(worker.generate_variations_for_fen.remote(
                fen,
                num_var=NUM_VARIATIONS_PER_FEN,
                min_len_var=MIN_LEN_VARIATION,
                max_len_var=MAX_LEN_VARIATION,
                top_k=TOP_K_MOVES_TO_CONSIDER,
                temperature=SOFTMAX_TEMPERATURE
            ))

        # Get results - this blocks until all tasks in the batch are complete
        try:
            batch_results = ray.get(tasks, timeout=120)  # 2 minute timeout per batch
        except Exception as e:
            logger.error(f"Error getting results from Ray for batch {i}: {e}")
            continue

        # Process results and update monitoring
        batch_processing_time = time.time() - batch_start_time
        valid_results = [res for res in batch_results if res.get("error") is None and res.get("variations")]
        errors = [res for res in batch_results if res.get("error") is not None]
        
        # Update performance monitoring
        monitor.record_batch_completion(len(fens), batch_processing_time, len(errors))
        
        # Record worker activity
        for result in batch_results:
            if result.get("worker_id") is not None and result.get("processing_time") is not None:
                monitor.record_worker_activity(result["worker_id"], result["processing_time"])

        # Save the results using data manager
        if valid_results:
            data_manager.add_batch_results(valid_results)
            total_variations = sum(len(res.get("variations", [])) for res in valid_results)
            monitor.metrics.total_variations_generated += total_variations

        # Log errors if any
        if errors:
            logger.warning(f"Batch {i} had {len(errors)} errors")
            for error in errors[:3]:  # Log first 3 errors
                logger.error(f"FEN: {error.get('fen', 'N/A')[:50]}..., Error: {error.get('error', 'Unknown')}")

        # Report performance stats periodically
        if i % STATS_REPORTING_INTERVAL == 0:
            stats = monitor.get_current_stats()
            logger.info(f"Performance Stats - Batch {i}: "
                       f"Throughput: {stats['throughput_per_second']:.2f} FENs/sec, "
                       f"Avg Worker Util: {stats['avg_worker_utilization']:.1f}%, "
                       f"Total Processed: {stats['total_fens_processed']}, "
                       f"CPU: {stats['system_cpu_percent']:.1f}%, "
                       f"Memory: {stats['system_memory_percent']:.1f}%")

    # Cleanup
    data_manager.flush()  # Write any remaining buffered data
    
    # Final stats
    final_stats = monitor.get_current_stats()
    logger.info("Processing completed!")
    logger.info(f"Final Stats: "
               f"Total FENs: {final_stats['total_fens_processed']}, "
               f"Total Batches: {final_stats['total_batches_processed']}, "
               f"Total Variations: {monitor.metrics.total_variations_generated}, "
               f"Total Errors: {final_stats['total_errors']}, "
               f"Runtime: {final_stats['total_runtime']:.2f}s, "
               f"Avg Throughput: {final_stats['throughput_per_second']:.2f} FENs/sec")

    # Shutdown Ray
    ray.shutdown()
    logger.info("Ray shut down")

def test_single_fen():
    """Test function to analyze a single FEN position and print variations."""
    test_fen = "8/4pp1p/3p1k1P/3P2p1/p1bq2P1/K4P2/2B5/1R6 b - - 5 37"
    print(f"Testing single FEN: {test_fen}")
    
    # Initialize Ray with just 1 worker for testing
    try:
        ray.init(num_cpus=1)
    except Exception as e:
        print(f"Ray initialization failed: {e}. Trying ignore_reinit_error=True")
        ray.init(num_cpus=1, ignore_reinit_error=True)

    print("Ray initialized for testing.")

    # Create single Stockfish worker
    print("Creating Stockfish worker...")
    worker = StockfishWorker.remote(STOCKFISH_BINARY_PATH, STOCKFISH_DEPTH, STOCKFISH_THREADS_PER_INSTANCE, 0)
    time.sleep(2)  # Allow worker to initialize
    print("Stockfish worker created.")

    # Generate variations for the test FEN
    try:
        result = ray.get(worker.generate_variations_for_fen.remote(
            test_fen,
            num_var=NUM_VARIATIONS_PER_FEN,
            min_len_var=MIN_LEN_VARIATION,
            max_len_var=MAX_LEN_VARIATION,
            top_k=TOP_K_MOVES_TO_CONSIDER,
            temperature=SOFTMAX_TEMPERATURE
        ))
        
        print(f"\nResults for FEN: {test_fen}")
        print(f"Error: {result.get('error', 'None')}")
        print(f"Processing time: {result.get('processing_time', 0):.3f}s")
        print(f"Worker ID: {result.get('worker_id', 'N/A')}")
        print(f"Number of variations: {len(result.get('variations', []))}")
        
        for i, variation in enumerate(result.get("variations", []), 1):
            print(f"Variation {i}: {' '.join(variation)}")
            
    except Exception as e:
        print(f"Error during test: {e}")
    
    # Shutdown Ray
    ray.shutdown()
    print("Ray shut down.")

if __name__ == "__main__":
    main()  # Run main function
    #test_single_fen()  # Uncomment to run small test instead