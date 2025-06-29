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


# --- Configuration ---
DIR_PATH = "/media/maxime/385e67e0-8703-4723-a0bc-af3a292fd030/ChessRL/data/compressed_pgns" # Or your actual path
STOCKFISH_BINARY_PATH = "stockfish/stockfish-ubuntu-x86-64-avx2" # Or your actual path - MAKE SURE THIS IS CORRECT
OUTPUT_DIR = "toremove" # Directory to save results
BATCH_SIZE = 32   # Increase batch size for better parallel utilization
NUM_WORKERS = os.cpu_count() or 4 # Use number of CPU cores, default 4 if undetectable
print(f"Using {NUM_WORKERS} workers.")
STOCKFISH_DEPTH = 15 # Minimum depth
# Threads per stockfish instance: Use 1 when running many parallel workers
STOCKFISH_THREADS_PER_INSTANCE = 1
# Parameters for variation generation
NUM_VARIATIONS_PER_FEN = 5
MIN_LEN_VARIATION = 3
MAX_LEN_VARIATION = 8
TOP_K_MOVES_TO_CONSIDER = 20 # Number of top moves to sample variations from
# Parameter for weighted sampling - higher value means more focus on top moves
# Lower value (~1) means more uniform distribution
SOFTMAX_TEMPERATURE = 1.0 # Adjust this to control the 'peakiness' of the distribution

# --- Helper Function for Score Conversion ---
# Define a large score for mate situations
MATE_SCORE = 3000 # Arbitrarily large centipawn value for mate

def score_to_numeric(score_dict, player_pov=True):
    """
    Converts Stockfish score {'Centipawn': cp, 'Mate': None} or {'Centipawn': None, 'Mate': m}
    into a single numeric value for comparison/probability calculation.
    Handles mate scores appropriately.
    player_pov=True means positive scores are good for the current player.
    """
    if score_dict.get("Mate") is not None:
        mate_in = score_dict["Mate"]
        if mate_in > 0: # Mate found for the current player
            # Higher score for faster mate
            return MATE_SCORE - mate_in
        elif mate_in < 0: # Mate found for the opponent
             # Lower score for faster opponent mate
            return -MATE_SCORE - mate_in # -mate_in makes it positive, so -30000 + 2 e.g.
        else: # Mate is 0? Should not happen in get_top_moves, treat as 0 cp
             return 0
    elif score_dict.get("Centipawn") is not None:
        return score_dict["Centipawn"]
    else:
        # Should not happen with valid get_top_moves output
        return 0

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
            self.stockfish.update_engine_parameters({
                # MultiPV > 1 needed for get_top_moves to return multiple moves
                # Set it high enough to cover TOP_K_MOVES_TO_CONSIDER
                "MultiPV": 1,
                "Threads": threads
            })
            # Optional: Add other parameters if needed
            # self.stockfish.update_engine_parameters({"Hash": 128})
            print(f"Stockfish worker initialized (PID: {os.getpid()}) with depth {depth}, threads {threads}, MultiPV {1}")
        except Exception as e:
            print(f"Error initializing Stockfish in worker: {e}")
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
            # Ensure MultiPV is 1 for getting single best move repeatedly
            # Store old MultiPV, set to 1, then restore
            # Note: This might add slight overhead. Consider if critical.
            # Alternatively, could use a separate stockfish command if library supports it.
            # current_multipv = self.stockfish.get_parameters()["MultiPV"] # Assumes get_parameters exists/works
            # self.stockfish.update_engine_parameters({"MultiPV": 1})
            # --- Simpler approach: rely on get_best_move using MultiPV=1 logic internally ---
            # The python-stockfish library's get_best_move seems independent of MultiPV setting.

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
             print(f"Error during Stockfish play: {e}. FEN: {self.stockfish.get_fen_position()}")
             moves.append(f"error: {e}")
        finally:
             # VERY IMPORTANT: Reset position back to the original FEN for the next variation
             self.stockfish.set_fen_position(initial_fen)
             # Optional: Restore MultiPV if changed above
             # self.stockfish.update_engine_parameters({"MultiPV": current_multipv}) # Assumes current_multipv was stored

        return moves

    def generate_variations_for_fen(self, fen, num_var=5, min_len_var=3, max_len_var=8, top_k=10, temperature=1.0):
        """
        Generates variations for a single FEN using this worker's Stockfish instance.
        Selects moves beyond the best one based on a probability distribution derived
        from their evaluation scores (using Softmax).
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

            # Get top moves - This uses the instance's depth setting and MultiPV setting
            best_first_moves_data = self.stockfish.get_top_moves(top_k)

            if not best_first_moves_data: # No legal moves from this position
                 return {"fen": fen, "variations": [], "error": "No legal moves"}

            # --- Weighted Sampling Logic ---
            # Ensure the absolute best move is always included
            best_move_data = best_first_moves_data[0]
            best_move = best_move_data["Move"]
            
            # Candidate moves for probabilistic sampling (all except the best)
            other_moves_data = best_first_moves_data[1:]

            first_moves = [best_move] # Start with the best move guaranteed

            # How many *additional* variations do we need?
            num_other_vars_needed = num_var - 1

            if num_other_vars_needed > 0 and other_moves_data:
                
                # Limit the number to sample by the available moves
                num_to_sample = min(num_other_vars_needed, len(other_moves_data))

                # Extract moves and calculate their numeric scores
                candidate_moves = [m["Move"] for m in other_moves_data]
                scores = np.array([score_to_numeric(m) for m in other_moves_data], dtype=float)
                scores = scores / 100
                # Apply Softmax to convert scores to probabilities
                # Higher temperature flattens the distribution, lower sharpens it
                # Subtract max score for numerical stability before exponentiating
                if len(scores) > 0 :
                    scores -= np.max(scores) # Stabilize softmax
                    exp_scores = np.exp(scores / temperature)
                    probabilities = exp_scores / np.sum(exp_scores)
                    
                    # Handle potential NaN if sum is zero (e.g., all scores were -inf)
                    if np.isnan(probabilities).any():
                         print(f"Warning: NaN encountered in probabilities for FEN {fen}. Using uniform distribution.")
                         probabilities = np.ones(len(candidate_moves)) / len(candidate_moves)

                    # Sample unique moves based on calculated probabilities
                    try:
                        sampled_indices = np.random.choice(
                            len(candidate_moves),
                            size=num_to_sample,
                            replace=False, # Ensure unique moves
                            p=probabilities
                        )
                    except:
                        print(candidate_moves, num_to_sample, probabilities)
                    sampled_moves = [candidate_moves[i] for i in sampled_indices]
                    first_moves.extend(sampled_moves)
                else: 
                    # Only one move was returned by get_top_moves, nothing else to sample
                    pass


            # --- End Weighted Sampling Logic ---

            # Generate lengths for each variation (for the *actual* number of variations we are generating)
            num_actual_vars = len(first_moves)
            # Generate continuation lengths (subtract 1 because first move is already chosen)
            var_lengths = np.random.randint(min_len_var -1 , max_len_var -1 +1, size=num_actual_vars) # +1 to include max_len_var-1


            results = []
            for i, (first_move, var_len) in enumerate(zip(first_moves, var_lengths)):
                current_variation = []
                current_variation.append(first_move)

                # Check if the first move is valid before proceeding (robustness)
                # We need to temporarily set the position, make the move, then generate continuation
                self.stockfish.set_fen_position(fen) # Reset to original state for this variation
                self.stockfish.make_moves_from_current_position([first_move])


                # Generate the rest of the variation using the *best* continuation moves
                # Note: _make_stockfish_play needs the FEN *after* the first move was made
                # fen_after_first_move = self.stockfish.get_fen_position() # Get state after making the move
                # BUT _make_stockfish_play RESETS to the FEN it's given. We need it to reset to the original FEN.
                # So, pass the original 'fen' and let _make_stockfish_play handle the sequence generation and final reset.

                # We already made the first move, so we need var_len more moves.
                if var_len > 0:
                    # Make the first move again WITHIN the context of this loop iteration
                    self.stockfish.set_fen_position(fen) # Reset
                    self.stockfish.make_moves_from_current_position([first_move]) # Make the specific first move
                    fen_after_first = self.stockfish.get_fen_position() # FEN for continuation

                    # Now generate continuation from *this* state, but reset to original 'fen' at the end
                    continuation_moves = self._make_stockfish_play(var_len, fen) # Pass original FEN for final reset
                    current_variation.extend(continuation_moves)
                # else: var_len was 0, meaning min_len_var was 1. Just keep the first move.

                results.append(current_variation)

                # Reset position for the *next* iteration of the 'first_moves' loop
                # This is crucial because _make_stockfish_play resets, but we need it clean for the next 'first_move'
                self.stockfish.set_fen_position(fen)


            return {"fen": fen, "variations": results, "error": None}

        except Exception as e:
            import traceback
            print(f"Error generating variations for FEN {fen} in worker {os.getpid()}: {e}")
            print(traceback.format_exc()) # Print full traceback for debugging
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
    try:
        ray.init(num_cpus=NUM_WORKERS)
    except Exception as e:
        print(f"Ray initialization failed: {e}. Trying ignore_reinit_error=True")
        ray.init(num_cpus=NUM_WORKERS, ignore_reinit_error=True) # Useful if running in interactive env

    print(f"Ray initialized with {NUM_WORKERS} workers.")

    # Create Stockfish worker actors
    print("Creating Stockfish workers...")
    workers = [StockfishWorker.remote(STOCKFISH_BINARY_PATH, STOCKFISH_DEPTH, STOCKFISH_THREADS_PER_INSTANCE)
               for _ in range(NUM_WORKERS)]
    # Small delay to allow workers to initialize (optional)
    time.sleep(2)
    print(f"{len(workers)} Stockfish workers created.")

    # Initialize the generator
    print(f"Initializing FEN generator from: {DIR_PATH} (using placeholder if path invalid)")
    try:
        # Attempt to use the real iterator
        from utils.parse import dir_iterator # Make sure this is importable
        fen_generator = dir_iterator(DIR_PATH, return_fen=True, batch_size=BATCH_SIZE, all_elo=True)
        print("--- Using real dir_iterator ---")
    except (ImportError, FileNotFoundError):
        # Fallback to placeholder if import fails or path is invalid
        print(f"Warning: Could not import or find path for real dir_iterator. Using placeholder.")
        fen_generator = dir_iterator(DIR_PATH, return_fen=True, batch_size=BATCH_SIZE, all_elo=True)


    # Main processing loop
    # Limit batches for testing, adjust as needed
    total_batches_to_process = 1000000 # Limit for demonstration
    print(f"Starting processing loop for up to {total_batches_to_process} batches...")
    start_time = time.time()
    batches_processed = 0

    for i in tqdm(range(total_batches_to_process), desc="Processing Batches"):
        try:
            fens = next(fen_generator)
            batches_processed += 1
        except StopIteration:
            print("\nFEN generator finished.")
            break
        except Exception as e:
             print(f"\nError getting next batch from generator: {e}")
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
                top_k=TOP_K_MOVES_TO_CONSIDER,
                temperature=SOFTMAX_TEMPERATURE # Pass the temperature parameter
            ))

        # Get results - this blocks until all tasks in the batch are complete
        try:
            batch_results = ray.get(tasks)
        except Exception as e:
            print(f"\nError getting results from Ray for batch {i}: {e}")
            # Handle potential Ray errors - maybe skip batch or try to recover
            continue # Skip saving this batch

        # Prepare data for saving (filter out potential errors if needed)
        valid_results = [res for res in batch_results if res.get("error") is None and res.get("variations")]
        errors = [res for res in batch_results if res.get("error") is not None]

        if errors:
             print(f"\nEncountered {len(errors)} errors in batch {i}:")
             #for err_info in errors[:5]: # Print first 5 errors
             #    print(f"  FEN: {err_info.get('fen', 'N/A')}, Error: {err_info.get('error', 'Unknown')}")

        # Save the results for the batch
        if valid_results: # Only save if there are valid results
            output_filename = os.path.join(OUTPUT_DIR, f"batch_{i}.pkl")
            try:
                with open(output_filename, "wb") as f:
                     pickle.dump(valid_results, f)
            except Exception as e:
                 print(f"\nError saving batch {i} to {output_filename}: {e}")
        elif not errors:
            # No errors, but no valid results either (e.g., all FENs had no legal moves)
            print(f"Batch {i} yielded no valid variations (and no errors).")


    end_time = time.time()
    print(f"\nProcessing finished.")
    print(f"Processed {batches_processed} batches.")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved in: {OUTPUT_DIR}")

    # Shutdown Ray
    ray.shutdown()
    print("Ray shut down.")