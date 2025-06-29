import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
try:
    from utils.vocab import policy_index
except ImportError:
    from vocab import policy_index

try:
    from utils.fen_encoder import fen_to_tensor
except ImportError:
    from fen_encoder import fen_to_tensor

import chess

# Precompute mapping from move string to index for fast lookup
_policy_map = {uci: idx for idx, uci in enumerate(policy_index)}
_pad_id = len(policy_index) - 1  # fallback padding ID
_PAD_FEN_STRING = "PAD_FEN"  # Placeholder for padded raw FENs

input_dir = "../stockfish_data/4a100/"

# Initialize a dummy tensor for FEN padding shape reference
# Ensure chess.Board().fen() provides a standard FEN for consistent shape.
try:
    _initial_board_fen = chess.Board().fen()
    _dummy_encoded_fen_tensor = torch.from_numpy(fen_to_tensor(_initial_board_fen))
    _zero_fen_pad_tensor = torch.zeros_like(_dummy_encoded_fen_tensor)
except Exception as e:
    print(f"Critical error initializing dummy FEN tensors: {e}")
    # Potentially raise an error or exit if this setup fails,
    # as it's crucial for padding.
    # For now, we'll let it proceed and it might error out later if these aren't set.
    _dummy_encoded_fen_tensor = None # Should cause errors if used
    _zero_fen_pad_tensor = None # Should cause errors if used


def encode_moves(moves_list):
    """
    Map UCI move strings to their indices, with fallback by stripping the last character.
    """
    ids = []
    for mov in moves_list:
        if mov in _policy_map:
            ids.append(_policy_map[mov])
        else:
            # Attempt to strip promotion char, otherwise use pad_id
            ids.append(_policy_map.get(mov[:-1], _pad_id))
    return torch.tensor(ids, dtype=torch.long)


def clip_and_batch(moves_tensors, clip):
    """
    Pad and/or clip a list of move-index tensors to shape (batch_size, clip).
    """
    padded = pad_sequence(moves_tensors, batch_first=True, padding_value=_pad_id)
    if padded.size(1) == clip: # Exact match
        return padded
    if padded.size(1) > clip: # Needs clipping
        return padded[:, :clip]
    # Needs padding
    extra = padded.new_full((padded.size(0), clip - padded.size(1)), _pad_id)
    return torch.cat([padded, extra], dim=1)


def batch_generator(input_dir, batch_size, block_size=64, return_fen=False, triple=False, device='cuda'):
    """
    Yield batches of encoded FENs and moves from parquet files.
    FENs are processed to (batch_size, block_size, *fen_to_tensor_shape).
    Moves are processed to (batch_size, block_size).
    
    Args:
        input_dir: Directory containing parquet files
        batch_size: Number of games per batch
        block_size: Number of moves/FENs per game (default: 5)
        return_fen: Whether to return raw FEN strings instead of tensors
        triple: Whether to return (tensors, moves, raw_fens, values)
        device: Device to place tensors on
    """
    if _zero_fen_pad_tensor is None:
        raise RuntimeError("FEN padding tensor initialization failed. Cannot proceed.")

    files = sorted([os.path.join(input_dir, f)
                    for f in os.listdir(input_dir) if f.endswith('.parquet')])
    #shuffle files 
    random.shuffle(files)
    game_fen_blocks_batch = []      # List of tensors, each (block_size, *fen_shape)
    game_move_sequences_batch = []  # List of tensors, each (<=block_size,) representing move indices
    raw_fens_for_output_batch = []  # List of lists of strings, if triple or return_fen
    value_batch = []

    
    for path in files:
        df = pd.read_parquet(path)
        for fen_initial, start_idx, moves_pgn_style,won_side in df.itertuples(index=False, name=None):
            won_side = won_side + 1 
            # 0 is black win, 1 is draw, 2 is white win
            board = chess.Board(fen_initial)
            
            current_game_fens_encoded_list = [] # Holds encoded FEN tensors for this game before stacking
            current_game_fens_raw_list = []     # Holds raw FEN strings for this game
            
        
            num_actual_moves_in_game = len(moves_pgn_style)
            # Determine how many FENs/moves to process for this game, up to block_size
            # Each move corresponds to a FEN *before* the move is made.
            # If we take K moves, we will have K FENs from which these moves were made.
            num_items_to_process = min(num_actual_moves_in_game, block_size)

            for i in range(num_items_to_process):
                
                current_fen_str = board.fen()
                current_game_fens_raw_list.append(current_fen_str)
                current_game_fens_encoded_list.append(torch.from_numpy(fen_to_tensor(current_fen_str)))
                try:
                    board.push_san(moves_pgn_style[i])
                except chess.IllegalMoveError:
                    # This move was illegal. We've already captured the FEN *before* this attempt.
                    # We will stop processing further moves for this game.
                    # The number of items processed will be `i`.
                    num_items_to_process = i 
                    break 
                except Exception as e: # Other board errors
                    # print(f"Warning: Error pushing move {moves_pgn_style[i]} from FEN {current_fen_str}: {e}")
                    num_items_to_process = i
                    break
            
            # Pad encoded FENs for the current game to block_size
            # current_game_fens_encoded_list contains 'num_items_to_process' items
            padding_needed_fens = block_size - len(current_game_fens_encoded_list)

            if padding_needed_fens > 0:
                value_batch.append(np.array([won_side]*(block_size-padding_needed_fens)+[3]*(padding_needed_fens)))
                current_game_fens_encoded_list.extend([_zero_fen_pad_tensor] * padding_needed_fens)
            else:
                value_batch.append(np.array([won_side]*block_size))
            # Stack the game's FENs into a single block tensor
            if not current_game_fens_encoded_list: # Should only happen if block_size is 0 or game had 0 moves from start
                 current_game_fen_block = torch.stack([_zero_fen_pad_tensor] * block_size)
            else:
                 current_game_fen_block = torch.stack(current_game_fens_encoded_list)
            
            game_fen_blocks_batch.append(current_game_fen_block)
            # Pad raw FENs (if needed for output)
            if return_fen or triple:
                # current_game_fens_raw_list has 'num_items_to_process' items
                padding_needed_raw_fens = block_size - len(current_game_fens_raw_list)
                if padding_needed_raw_fens > 0:
                    current_game_fens_raw_list.extend([_PAD_FEN_STRING] * padding_needed_raw_fens)
                raw_fens_for_output_batch.append(current_game_fens_raw_list)
            
            # Process moves for the current game: take moves corresponding to the processed FENs, then encode.
            # These are the moves that *were* successfully processed or were intended to be processed.
            actual_moves_for_game = moves_pgn_style[:num_items_to_process]
            encoded_game_moves = encode_moves(actual_moves_for_game) # 1D tensor of move indices
            game_move_sequences_batch.append(encoded_game_moves)
            if len(game_fen_blocks_batch) == batch_size:
                yield from _yield_batch(
                    game_fen_blocks_batch,
                    game_move_sequences_batch,
                    raw_fens_for_output_batch if (return_fen or triple) else None,
                    return_fen,
                    triple,
                    device,
                    value_batch,
                    block_size
                )
                game_fen_blocks_batch.clear()
                game_move_sequences_batch.clear()
                if return_fen or triple:
                    raw_fens_for_output_batch.clear()
                value_batch.clear()
    if game_fen_blocks_batch:
        yield from _yield_batch(
            game_fen_blocks_batch,
            game_move_sequences_batch,
            raw_fens_for_output_batch if (return_fen or triple) else None,
            return_fen,
            triple,
            device,
            value_batch,
            block_size
        )


def _yield_batch(
        fen_blocks_list,         # List of [block_size, *fen_shape] tensors
        move_sequences_list,     # List of [<=block_size] move index tensors
        raw_fens_list_of_lists,  # List of lists of FEN strings (each inner list block_size long), or None
        return_fen_flag,
        triple_flag,
        device,
        value_batch,
        block_size
    ):
    if return_fen_flag:
        # raw_fens_list_of_lists is expected to be a list where each item is 
        # a list of block_size FEN strings (actual or _PAD_FEN_STRING)
        yield raw_fens_list_of_lists
    else:
        fens_batch_tensor = torch.stack(fen_blocks_list).to(device)
        # moves_batch_tensor will be (batch_size, block_size) after clip_and_batch
        moves_batch_tensor = clip_and_batch(move_sequences_list, clip=block_size).to(device)
        value_numpy = np.array(value_batch)
        value_batch_tensor = torch.from_numpy(value_numpy).to(device).to(torch.int64)
        if triple_flag:
            # raw_fens_list_of_lists is the batch of FEN strings
            yield fens_batch_tensor, moves_batch_tensor, raw_fens_list_of_lists, value_batch_tensor
        else:
            yield fens_batch_tensor, moves_batch_tensor, value_batch_tensor


if __name__ == '__main__':
    from tqdm import tqdm


    # Test 3: Triple (FEN tensors, Move tensors, FEN strings)
    print("\nTesting triple=True:")
    try:
        gen_triple = batch_generator(input_dir, batch_size=32, block_size=1, triple=True, device=('cuda' if torch.cuda.is_available() else 'cpu'))
        for i in tqdm(range(2)):
            f_batch_t, m_batch_t, raw_fen_batch_t,value_batch_t = next(gen_triple)
            print(f"Batch {i+1} (Triple):")
            print(f"  FENs tensor shape: {f_batch_t.shape}, Moves tensor shape: {m_batch_t.shape}")
            if isinstance(raw_fen_batch_t, list) and len(raw_fen_batch_t) > 0:
                print(f"  Number of games in raw FENs batch: {len(raw_fen_batch_t)}")
                print(f"  Number of FENs in first game (raw): {len(raw_fen_batch_t[0])}")
                print(f"  Value batch tensor shape: {value_batch_t.shape}")
                print(f"  First FEN string from first game (raw): {raw_fen_batch_t[0][0]}")
                print(f"  Value batch tensor: {value_batch_t[0]}")
            else:
                print(f"  Received unexpected data format for raw FENs in triple: {type(raw_fen_batch_t)}")
    except Exception as e:
        print(f"Error during triple test: {e}")

    # Example of running through more batches to check for exhaustion or errors
    # print("\nRunning through 100 batches (standard):")
    # gen = batch_generator(input_dir, batch_size=32, block_size=5, device=('cuda' if torch.cuda.is_available() else 'cpu'))
    # for _ in tqdm(range(100), desc="Processing batches"):
    #     try:
    #         batch_data = next(gen)
    #         # Basic check, more can be added
    #         if not (isinstance(batch_data, tuple) and len(batch_data) == 2):
    #             print("Batch data format error!")
    #             break
    #         if batch_data[0].shape[1] != block_size or batch_data[1].shape[1] != block_size:
    #             print(f"Dimension mismatch! FENs: {batch_data[0].shape}, Moves: {batch_data[1].shape}")
    #             break
    #     except StopIteration:
    #         print("Generator exhausted.")
    #         break
    #     except Exception as e:
    #         print(f"Error during extended run: {e}")
    #         break
    print("\nTests complete.")
