import os
import pickle
import random
import torch
import numpy as np
import chess
import pandas as pd
# Ensure utils. Sibling modules are correctly found if running as script
import sys
if '.' not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    from utils.vocab import policy_index
    from utils.parse import clip_and_batch, encode_moves_bis
    from utils.fen_encoder import fen_to_tensor
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Ensure that 'utils' directory is in PYTHONPATH or script is run from project root.")
    # Fallback for direct script run if utils are in ../utils
    try:
        from ..utils.vocab import policy_index
        from ..utils.parse import clip_and_batch, encode_moves_bis
        from ..utils.fen_encoder import fen_to_tensor
    except ImportError:
        raise e # Re-raise if fallback fails

from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 64 # Max sequence length for tokens and FEN tensors
PAD_ID = policy_index.index("padding_token") if "padding_token" in policy_index else len(policy_index) -1


# Special FEN Tensors (will be initialized once)
_ZERO_FEN_PAD_TENSOR = None
_MINUS_ONE_FEN_TENSOR = None
_PLUS_ONE_FEN_TENSOR = None
_DUMMY_FEN_SHAPE = None

def _initialize_special_fen_tensors():
    """Initializes special FEN tensors based on a dummy FEN."""
    global _ZERO_FEN_PAD_TENSOR, _MINUS_ONE_FEN_TENSOR, _PLUS_ONE_FEN_TENSOR, _DUMMY_FEN_SHAPE
    if _ZERO_FEN_PAD_TENSOR is not None: # Already initialized
        return

    try:
        # Use a standard initial FEN to get tensor shape
        dummy_board = chess.Board()
        dummy_fen_np = fen_to_tensor(dummy_board.fen())
        dummy_fen_tensor = torch.from_numpy(dummy_fen_np).to(DEVICE)
        _DUMMY_FEN_SHAPE = dummy_fen_tensor.shape

        _ZERO_FEN_PAD_TENSOR = torch.zeros_like(dummy_fen_tensor)
        _MINUS_ONE_FEN_TENSOR = torch.full_like(dummy_fen_tensor, -1)
        _PLUS_ONE_FEN_TENSOR = torch.ones_like(dummy_fen_tensor)
    except Exception as e:
        print(f"Critical error initializing special FEN tensors: {e}")
        raise RuntimeError("Failed to initialize special FEN tensors. Cannot proceed.")

_initialize_special_fen_tensors() # Initialize on module load


files = []


data_path = "/home/maxime/cot/cot"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))

data_path = "./cot"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))
random.shuffle(files)

def _construct_token_sequence(game_vars):
    """
    Constructs the full token sequence for a game from its variations.
    Variations are shuffled. If a variation ends with 'end', it's truncated.
    Sequence: <thinking>, moves_var1, end_variation, ..., </thinking>, first_move, end
    """
    if not game_vars or not game_vars[0]: # Ensure there's at least one variation and it's not empty
        return [], None # Not enough data to form a sequence

    # first_move_played is the first move of the *first variation in the original list*
    # This should be determined *before* shuffling, if it's meant to be the absolute first move.
    # Assuming game_vars[0][0] is robust enough (i.e., first var always has at least one move)
    try:
        first_move_played = game_vars[0][0] # Assumes UCI format
    except (IndexError, TypeError):
        # print(f"Warning: Could not determine first_move_played from game_vars: {game_vars}. Skipping this logic for the game.")
        return [], None # Or handle as an error / skip game

    processed_tokens = ["<thinking>"]
    
    # Shuffle a copy of the variations list to avoid modifying the original if it's used elsewhere
    shuffled_game_vars = list(game_vars)
    random.shuffle(shuffled_game_vars)
    
    for var_moves_list in shuffled_game_vars:
        if isinstance(var_moves_list, list):
            # Truncate if the last token of this specific variation is 'end'
            current_var_tokens = list(var_moves_list) # Work with a copy
            if current_var_tokens and current_var_tokens[-1] == "end":
                current_var_tokens.pop()
            
            if current_var_tokens: # Add only if not empty after potential truncation
                processed_tokens.extend(current_var_tokens)
        else: 
            # This case implies game_vars was not a list of lists, but a flat list of moves.
            # If so, shuffling individual moves and then adding 'end_variation' after each might not be intended.
            # Based on previous structure, game_vars is expected to be a list of variations (lists of moves).
            # If var_moves_list is a single string (a move), we add it.
            # The 'end' truncation logic wouldn't apply here unless it's the only token.
            if var_moves_list == "end": # A single 'end' token as a variation.
                pass # Skip it if it's by itself after truncation logic for lists
            else:
                processed_tokens.append(var_moves_list)
        
        processed_tokens.append("end_variation")
    
    processed_tokens.append("</thinking>")
    processed_tokens.append(first_move_played) # The pre-determined first move
    processed_tokens.append("end")
    return processed_tokens, first_move_played


def _generate_fen_sequence_for_game(initial_fen_str, game_tokens_no_end):
    """
    Generates a sequence of FEN tensors corresponding to game_tokens (excluding 'end').
    Returns (list_of_fen_tensors, success_flag).
    On illegal move, success_flag is False.
    """
    fen_tensors_for_game = []
    board = chess.Board()
    original_board_sim = chess.Board(initial_fen_str) # For resets

    try:
        board.set_fen(initial_fen_str)
    except ValueError as e:
        print(f"Warning: Invalid initial FEN '{initial_fen_str}': {e}. Skipping game.")
        return None, False

    #start by adding initial fen
    current_fen_tensor = torch.from_numpy(fen_to_tensor(initial_fen_str)).to(DEVICE)
    fen_tensors_for_game.append(current_fen_tensor)

    for token in game_tokens_no_end:
        if token == "<thinking>":
            fen_tensors_for_game.append(_MINUS_ONE_FEN_TENSOR.clone())
            board.set_fen(original_board_sim.fen()) # Reset board state
        elif token == "</thinking>":
            fen_tensors_for_game.append(_PLUS_ONE_FEN_TENSOR.clone())
            board.set_fen(original_board_sim.fen()) # Reset board state
        elif token == "end_variation":
            fen_tensors_for_game.append(_ZERO_FEN_PAD_TENSOR.clone()) # Use zero tensor for end_variation
            board.set_fen(original_board_sim.fen()) # Reset board state
        else: # It's a chess move (assumed UCI)
            try:
                board.push_uci(token)
                current_fen_tensor = torch.from_numpy(fen_to_tensor(board.fen())).to(DEVICE)
                fen_tensors_for_game.append(current_fen_tensor)
            except chess.InvalidMoveError: # Raised for ill-formed UCI
                 print(f"Warning: Ill-formed UCI move '{token}' from FEN '{board.fen()}' in game '{initial_fen_str}'. Discarding game.")
                 return None, False
            except chess.IllegalMoveError:
                print(f"Warning: Illegal move '{token}' from FEN '{board.fen()}' in game '{initial_fen_str}'. Discarding game.")
                return None, False
            except Exception as e: # Catch other potential errors from chess library
                print(f"Warning: Error processing move '{token}' from FEN '{board.fen()}' (Game: '{initial_fen_str}'): {e}. Discarding game.")
                return None, False        
    return fen_tensors_for_game, True


def gen(batch_size=32):
    """
    Generator yielding batches of FEN sequences, token sequences, and initial FEN strings.
    Output format: (batch_fen_sequences, batch_token_sequences, batch_initial_fens_str)
    batch_fen_sequences: Tensor (batch_size, BLOCK_SIZE, *fen_dims)
    batch_token_sequences: Tensor (batch_size, BLOCK_SIZE) (indices)
    batch_initial_fens_str: List of strings (batch_size)
    """
    _initialize_special_fen_tensors() # Ensure tensors are ready

    batch_fen_tensor_lists = []         # Collects lists of FEN tensors for each game
    batch_raw_token_lists = []          # Collects lists of token strings for each game
    batch_initial_fens_str_list = []

    for file_path in files:
        # print(f"Processing file: {file_path}")
        try:
            df = pd.read_parquet(file_path)
            #group by fen
            df = df.groupby("fen")
        except Exception as e:
            print(f"Warning: Could not load or process file {file_path}: {e}")
            continue
        for initial_fen, group in df:
            game_vars = group["variation"].tolist()
            game_vars = [var.split(" ") for var in game_vars]
            if not initial_fen or not game_vars:
                # print(f"Warning: Missing FEN or variations in {file_path}. Skipping game.")
                continue
            processed_game_tokens, _ = _construct_token_sequence(game_vars)

            if not processed_game_tokens:
                # print(f"Warning: Could not construct token sequence for game with FEN {initial_fen}. Skipping.")
                continue
            
            if len(processed_game_tokens) == 0 or len(processed_game_tokens) > BLOCK_SIZE:
                # print(f"Info: Game with FEN {initial_fen} has {len(processed_game_tokens)} tokens. Exceeds BLOCK_SIZE {BLOCK_SIZE} or is empty. Discarding.")
                continue
            # Generate FENs for tokens up to (but not including) the 'end' token
            game_fen_tensors_list, success = _generate_fen_sequence_for_game(initial_fen, processed_game_tokens[:-1])
            assert len(game_fen_tensors_list) == len(processed_game_tokens)
            if not success:
                # Warning already printed by _generate_fen_sequence_for_game
                continue
            
            batch_fen_tensor_lists.append(game_fen_tensors_list)
            batch_raw_token_lists.append(processed_game_tokens) # Store the full token list (with 'end')
            batch_initial_fens_str_list.append(initial_fen)

            if len(batch_fen_tensor_lists) == batch_size:
                # Prepare FEN tensor batch
                padded_fen_sequences_for_batch = []
                for fen_list in batch_fen_tensor_lists:
                    padding_needed = BLOCK_SIZE - len(fen_list)
                    # Pad with _ZERO_FEN_PAD_TENSOR. fen_list already contains tensors on DEVICE.
                    padded_list = fen_list + [_ZERO_FEN_PAD_TENSOR.clone() for _ in range(padding_needed)]
                    padded_fen_sequences_for_batch.append(torch.stack(padded_list))
                final_batch_fens = torch.stack(padded_fen_sequences_for_batch) # Already on DEVICE

                # Prepare token tensor batch
                # encode_moves_bis expects list of strings, returns tensor.
                # clip_and_batch expects list of tensors.
                encoded_tokens_for_batch = [encode_moves_bis(str_list) for str_list in batch_raw_token_lists]
                final_batch_tokens = clip_and_batch(encoded_tokens_for_batch, clip=BLOCK_SIZE).to(DEVICE)
                
                yield final_batch_fens, final_batch_tokens, batch_initial_fens_str_list
                
                batch_fen_tensor_lists.clear()
                batch_raw_token_lists.clear()
                batch_initial_fens_str_list.clear()

    # Yield any remaining partial batch
    if batch_fen_tensor_lists:
        padded_fen_sequences_for_batch = []
        for fen_list in batch_fen_tensor_lists:
            padding_needed = BLOCK_SIZE - len(fen_list)
            padded_list = fen_list + [_ZERO_FEN_PAD_TENSOR.clone() for _ in range(padding_needed)]
            padded_fen_sequences_for_batch.append(torch.stack(padded_list))
        final_batch_fens = torch.stack(padded_fen_sequences_for_batch)

        encoded_tokens_for_batch = [encode_moves_bis(str_list) for str_list in batch_raw_token_lists]
        final_batch_tokens = clip_and_batch(encoded_tokens_for_batch, clip=BLOCK_SIZE).to(DEVICE)
        
        yield final_batch_fens, final_batch_tokens, batch_initial_fens_str_list

def gen2(batch_size_multiplier=8, sub_batch_size=32):
    """
    Aggregates multiple batches from gen() into larger batches.
    """
    sub_generator = gen(batch_size=sub_batch_size)
    while True:
        batch_fens_list = []
        batch_tokens_list = []
        aggregated_initial_fens_str = []
        
        current_aggregated_size = 0
        target_batch_size = sub_batch_size * batch_size_multiplier

        try:
            for _ in range(batch_size_multiplier): # Try to get `batch_size_multiplier` sub-batches
                f_sub_batch, t_sub_batch, initial_fens_sub_batch = next(sub_generator)
                batch_fens_list.append(f_sub_batch)
                batch_tokens_list.append(t_sub_batch)
                aggregated_initial_fens_str.extend(initial_fens_sub_batch)
                current_aggregated_size += f_sub_batch.shape[0]
                if current_aggregated_size >= target_batch_size: # Should not happen if sub_batch_size is consistent
                    break
            
            if not batch_fens_list: # Sub-generator exhausted before filling one sub-batch
                 return

            final_fens = torch.cat(batch_fens_list, dim=0)
            final_tokens = torch.cat(batch_tokens_list, dim=0)
            
            yield final_fens, final_tokens, aggregated_initial_fens_str

        except StopIteration:
            # If sub_generator is exhausted, yield any partially aggregated batch and then stop
            if batch_fens_list:
                final_fens = torch.cat(batch_fens_list, dim=0)
                final_tokens = torch.cat(batch_tokens_list, dim=0)
                yield final_fens, final_tokens, aggregated_initial_fens_str
            return


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"BLOCK_SIZE: {BLOCK_SIZE}, PAD_ID: {PAD_ID}")
    print(f"Special FEN Tensors Initialized: Zero tensor shape: {_ZERO_FEN_PAD_TENSOR.shape if _ZERO_FEN_PAD_TENSOR is not None else 'Not Init'}")

    # Create a dummy data file for testing if no real data is available
    # This helps in environments where data paths might not be set up
    if not files:
        print("No data files found by script. Creating dummy data for testing.")
        dummy_data_content = {
            "fens": ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"],
            "var": [ # Variations for each FEN
                [["e2e4", "e7e5"], ["g1f3", "b8c6"]], # Vars for FEN 1
                [["g1f3", "g8f6"], ["d2d4"]]  # Vars for FEN 2
            ]
        }
        dummy_file_path = "dummy_test_data.pkl"
        with open(dummy_file_path, "wb") as df:
            pickle.dump(dummy_data_content, df)
        files.append(dummy_file_path)
        print(f"Created and using {dummy_file_path} for testing.")


    print("\nTesting gen() generator:")
    try:
        # Test with a small batch size
        g = gen(batch_size=32) 
        
        # Get a few batches
        for i in range(1):
            #print(f"\n--- Batch {i+1} ---")
            try:
                batch_fens, batch_tokens, batch_initial_fens = next(g)

                print(f"  Initial FENs in batch: {len(batch_initial_fens)}")
                if batch_initial_fens:
                    print(f"    Example initial FEN: {batch_initial_fens[0]}")

                print(f"  FEN sequences tensor shape: {batch_fens.shape}")
                print(f"  FEN sequences tensor dtype: {batch_fens.dtype}")
                print(f"  FEN sequences tensor device: {batch_fens.device}")

                print(f"  Token sequences tensor shape: {batch_tokens.shape}")
                print(f"  Token sequences tensor dtype: {batch_tokens.dtype}")
                print(f"  Token sequences tensor device: {batch_tokens.device}")

                if batch_tokens.numel() > 0:
                    print(f"    Example token sequence (indices): {batch_tokens[0, :10]}...") # Print first 10 tokens
                    
                    # Decode first example token sequence for readability
                    if policy_index:
                        decoded_tokens = [policy_index[idx.item()] if idx.item() < len(policy_index) else "UNK_PAD" for idx in batch_tokens[0]]
                        print(f"    Decoded example tokens: {' '.join(decoded_tokens)}")

            except StopIteration:
                print("Generator exhausted.")
                break
            except Exception as e_inner:
                print(f"Error getting batch {i+1}: {e_inner}")
                import traceback
                traceback.print_exc()
                break
        
        # Clean up dummy file if created
        if "dummy_test_data.pkl" in files:
            os.remove("dummy_test_data.pkl")

    except Exception as e:
        print(f"Error during gen() test: {e}")
        import traceback
        traceback.print_exc()


    print("\nScript finished.")