import os
import ray
import time
import random
import pickle
import numpy as np
from tqdm import tqdm
from stockfish import Stockfish
# Assuming utils.parse.dir_iterator exists as in the original code
from utils.parse import dir_iterator 

# Placeholder for dir_iterator if you don't have the actual file
# This is just for the code to be runnable for demonstration.
# Replace with your actual import.

# --- Configuration ---
DIR_PATH = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/casual_pgn" # Or your actual path
STOCKFISH_BINARY_PATH = "stockfish/stockfish-ubuntu-x86-64-avx2" # Or your actual path
OUTPUT_DIR = "toremove" # Directory to save results
BATCH_SIZE = 32  # Increase batch size for better parallel utilization
NUM_WORKERS = os.cpu_count()  # Use number of CPU cores as a starting point
print(NUM_WORKERS)
STOCKFISH_DEPTH = 15 # Minimum depth
# Threads per stockfish instance: Use 1 when running many parallel workers
STOCKFISH_THREADS_PER_INSTANCE = 1 
# Parameters for variation generation
NUM_VARIATIONS_PER_FEN = 5
MIN_LEN_VARIATION = 3
MAX_LEN_VARIATION = 8
TOP_K_MOVES_TO_CONSIDER = 10 # Number of top moves to sample variations from

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Ray Actor Definition ---
@ray.remote
class StockfishWorker:
    def __init__(self, stockfish_path, depth, threads):
        try:
            # Each worker gets its own Stockfish instance
            self.stockfish = Stockfish(path=stockfish_path, depth=depth)
            # Set parameters like Threads for this specific instance
            # Use lower thread count per instance when running many workers
            self.stockfish.update_engine_parameters({
                "MultiPV": 1, # Keep MultiPV=1; get_top_moves handles finding multiple moves
                "Threads": threads 
            })
            # Optional: Add other parameters if needed
            # self.stockfish.update_engine_parameters({"Hash": 128}) 
            print(f"Stockfish worker initialized (PID: {os.getpid()}) with depth {depth}, threads {threads}")
        except Exception as e:
            print(f"Error initializing Stockfish in worker: {e}")
            self.stockfish = None # Mark as invalid

    def _make_stockfish_play(self, k, initial_fen):
        """
        Generates k moves starting from the current position.
        Resets to the initial_fen afterwards.
        (Helper method internal to the actor)
        """
        if not self.stockfish: return ["error: no stockfish"]
        
        moves = []
        try:
            for _ in range(k):
                move = self.stockfish.get_best_move() # This uses the instance's depth setting
                if move is None: # No legal moves (checkmate/stalemate)
                    move = "end" 
                    moves.append(move)
                    break 
                moves.append(move)
                self.stockfish.make_moves_from_current_position([move])
        except Exception as e:
             print(f"Error during Stockfish play: {e}. FEN: {self.stockfish.get_fen_position()}")
             moves.append(f"error: {e}")
        finally:
             # VERY IMPORTANT: Reset position back to the original FEN for the next variation
             self.stockfish.set_fen_position(initial_fen) 
        return moves

    def generate_variations_for_fen(self, fen, num_var=5, min_len_var=3, max_len_var=8, top_k=20):
        """
        Generates variations for a single FEN using this worker's Stockfish instance.
        """
        if not self.stockfish:
             return {"fen": fen, "variations": [], "error": "Stockfish not initialized"}
        
        # Validate FEN first
        if not self.stockfish.is_fen_valid(fen):
            print(f"Worker {os.getpid()} received invalid FEN: {fen}")
            return {"fen": fen, "variations": [], "error": "Invalid FEN"}

        variations = []
        try:
            self.stockfish.set_fen_position(fen)
            
            # Get top moves - This uses the instance's depth setting
            # Ensure MultiPV is sufficient if get_top_moves relies on it, 
            # or confirm the library handles it internally. Let's assume top_k works.
            # If issues arise, consider setting MultiPV = top_k before this call.
            best_first_moves = self.stockfish.get_top_moves(top_k)

            if not best_first_moves: # No legal moves from this position
                 return {"fen": fen, "variations": [], "error": "No legal moves"}

            # Separate the best move
            best_move_data = best_first_moves.pop(0)
            best_move = best_move_data["Move"]

            # Sample other moves for variations
            num_other_vars_max = min(num_var - 1, len(best_first_moves))
            other_vars_data = random.sample(best_first_moves, num_other_vars_max)
            other_moves = [move["Move"] for move in other_vars_data]
            
            first_moves = [best_move] + other_moves
            
            # Generate lengths for each variation
            num_actual_vars = len(first_moves)
            var_lengths = np.random.randint(min_len_var - 1, max_len_var, size=num_actual_vars)

            results = []
            for i, (first_move, var_len) in enumerate(zip(first_moves, var_lengths)):
                current_variation = []
                current_variation.append(first_move)
                
                # Make the first move
                self.stockfish.make_moves_from_current_position([first_move])
                
                # Generate the rest of the variation
                continuation_moves = self._make_stockfish_play(var_len, fen) # Pass original FEN for reset
                current_variation.extend(continuation_moves)
                results.append(current_variation)
                
                # Note: _make_stockfish_play should reset the position back to 'fen'
                # If it didn't, we would need self.stockfish.set_fen_position(fen) here.

            return {"fen": fen, "variations": results, "error": None}

        except Exception as e:
            print(f"Error generating variations for FEN {fen} in worker {os.getpid()}: {e}")
            # Attempt to reset state in case of error
            try:
                self.stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") # Reset to start
            except:
                pass # Ignore error during reset attempt
            return {"fen": fen, "variations": [], "error": str(e)}


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting script...")
    # Initialize Ray
    # Include object_store_memory if needed and you have RAM, can speed up data transfer
    # ray.init(num_cpus=NUM_WORKERS, object_store_memory=10**9) # Example: 1 GB
    ray.init(num_cpus=NUM_WORKERS) 
    print(f"Ray initialized with {NUM_WORKERS} workers.")

    # Create Stockfish worker actors
    print("Creating Stockfish workers...")
    workers = [StockfishWorker.remote(STOCKFISH_BINARY_PATH, STOCKFISH_DEPTH, STOCKFISH_THREADS_PER_INSTANCE) 
               for _ in range(NUM_WORKERS)]
    print(f"{len(workers)} Stockfish workers created.")

    # Initialize the generator
    print(f"Initializing FEN generator from: {DIR_PATH}")
    # Use your actual dir_iterator here
    fen_generator = dir_iterator(DIR_PATH, return_fen=True, batch_size=BATCH_SIZE, all_elo=True) 

    # Main processing loop
    total_batches_to_process = 100000 # Limit for demonstration, set as needed (e.g., 1,000,000)
    print(f"Starting processing loop for {total_batches_to_process} batches...")
    start_time = time.time()

    for i in tqdm(range(total_batches_to_process), desc="Processing Batches"):
        try:
            fens = next(fen_generator)
        except StopIteration:
            print("FEN generator finished.")
            break
            
        if not fens:
             print("Received empty batch, stopping.")
             break

        # Distribute tasks to workers
        tasks = []
        for idx, fen in enumerate(fens):
            # Assign task to a worker in round-robin fashion
            worker = workers[idx % NUM_WORKERS] 
            tasks.append(worker.generate_variations_for_fen.remote(
                fen, 
                num_var=NUM_VARIATIONS_PER_FEN,
                min_len_var=MIN_LEN_VARIATION,
                max_len_var=MAX_LEN_VARIATION,
                top_k=TOP_K_MOVES_TO_CONSIDER
            ))

        # Get results - this blocks until all tasks in the batch are complete
        batch_results = ray.get(tasks)

        # Prepare data for saving (filter out potential errors if needed)
        # The result now includes the FEN and any errors
        valid_results = [res for res in batch_results if res.get("error") is None]
        errors = [res for res in batch_results if res.get("error") is not None]
        
        if errors:
             print(f"\nEncountered {len(errors)} errors in batch {i}:")
             #for err_info in errors[:5]: # Print first 5 errors
             #   print(f"  FEN: {err_info['fen']}, Error: {err_info['error']}")

        # Save the results for the batch
        output_filename = os.path.join(OUTPUT_DIR, f"batch_{i}.pkl")
        try:
             # Save only the valid results, or save the full batch_results
             # depending on whether you want to keep error info
             # Saving structure: List of dictionaries [{"fen": fen, "variations": [...]}, ...]
             with open(output_filename, "wb") as f:
                 pickle.dump(valid_results, f) 
                 # Or: pickle.dump(batch_results, f) # To include errors
        except Exception as e:
             print(f"\nError saving batch {i} to {output_filename}: {e}")

    end_time = time.time()
    print(f"\nProcessing finished.")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved in: {OUTPUT_DIR}")

    # Shutdown Ray
    ray.shutdown()
    print("Ray shut down.")