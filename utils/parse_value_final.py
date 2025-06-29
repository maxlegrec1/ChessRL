import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from collections import deque
import pyarrow.parquet as pq
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

input_dir = "/mnt/2tb/LeelaDataReader/output_parquet_final"

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


def get_padding_move():
    """
    Return the padding move index.
    """
    return _pad_id


def mirror_fen(fen):
    """
    Mirror the FEN to change perspective from white to black and vice versa.
    
    Args:
        fen: Original FEN string
        
    Returns:
        tuple: (mirrored_fen)
    """
    def swap_case(char):
        """Swap case of chess piece characters"""
        if char.isupper():
            return char.lower()
        elif char.islower():
            return char.upper()
        else:
            return char
    
    parts = fen.split(' ')
    if len(parts) != 6:
        # Malformed FEN, return original
        return fen
    
    # 1. Mirror the board position (flip vertically and swap piece colors)
    board_part = parts[0]
    ranks = board_part.split('/')
    # Reverse the ranks (rank 8 becomes rank 1, etc.) and swap piece colors
    mirrored_ranks = []
    for rank in reversed(ranks):
        mirrored_rank = ''.join(swap_case(char) for char in rank)
        mirrored_ranks.append(mirrored_rank)
    mirrored_board = '/'.join(mirrored_ranks)
    
    # 2. Swap active color
    active_color = 'b' if parts[1] == 'w' else 'w'
    
    # 3. Swap castling rights (KQkq -> kqKQ)
    castling = parts[2]
    if castling == '-':
        mirrored_castling = '-'
    else:
        # Separate white and black castling rights and swap them
        white_castling = ''.join(c for c in castling if c.isupper())
        black_castling = ''.join(c for c in castling if c.islower())
        # Swap: white becomes black (lowercase) and black becomes white (uppercase)
        mirrored_castling = black_castling.upper() + white_castling.lower()
        if not mirrored_castling:
            mirrored_castling = '-'
    
    # 4. Mirror en passant target square
    en_passant = parts[3]
    if en_passant == '-':
        mirrored_en_passant = '-'
    else:
        # Flip the rank: rank 6 -> rank 3, rank 3 -> rank 6
        file = en_passant[0]  # a-h stays the same
        rank = en_passant[1]  # flip rank
        if rank == '6':
            mirrored_rank = '3'
        elif rank == '3':
            mirrored_rank = '6'
        else:
            mirrored_rank = rank  # shouldn't happen for valid en passant
        mirrored_en_passant = file + mirrored_rank
    
    # 5. Keep halfmove and fullmove clocks the same
    halfmove_clock = parts[4]
    fullmove_number = parts[5]
    
    # Reconstruct the mirrored FEN
    mirrored_fen = f"{mirrored_board} {active_color} {mirrored_castling} {mirrored_en_passant} {halfmove_clock} {fullmove_number}"
  
    return mirrored_fen


def process_probabilities(fen, white_win, draw_prob, black_win,q_win,q_draw,q_loss):
    """
    Convert probabilities to winning side index.
    
    Args:
        fen: FEN string to determine whose turn it is
        white_win: Probability that white wins
        draw_prob: Probability of draw
        black_win: Probability that black wins
        
    Returns:
        int: Index of the outcome (0=black wins, 1=draw, 2=white wins)
    """
    return np.array([black_win, draw_prob, white_win]),np.array([q_loss,q_draw,q_win])


def batch_generator(input_dir, batch_size, block_size=1, return_fen=False, triple=False, device='cuda'):
    """
    Yield batches of encoded FENs and padding moves from parquet files.
    Load files row group by row group for memory efficiency.
    
    Args:
        input_dir: Directory containing parquet files
        batch_size: Number of positions per batch
        block_size: Always 1 for this version
        return_fen: Whether to return raw FEN strings instead of tensors
        triple: Whether to return (tensors, moves, raw_fens, values)
        device: Device to place tensors on
    """
    if _zero_fen_pad_tensor is None:
        raise RuntimeError("FEN padding tensor initialization failed. Cannot proceed.")

    # Force block_size to be 1
    block_size = 1

    files = sorted([os.path.join(input_dir, f)
                    for f in os.listdir(input_dir) if f.endswith('.parquet')])
    
    # Process files sequentially
    for file_idx, file_path in enumerate(files):
        print(f"Loading file {file_idx + 1}/{len(files)}: {file_path}")
        
        try:
            parquet_file = pq.ParquetFile(file_path)
        except Exception as e:
            print(f"Error opening parquet file {file_path}: {e}")
            continue
            
        num_row_groups = parquet_file.num_row_groups
        print(f"File has {num_row_groups} row groups")
        
        # Process each row group
        for row_group_idx in range(num_row_groups):
            try:
                # Read one row group at a time
                row_group_df = parquet_file.read_row_group(row_group_idx).to_pandas()
                print(f"Processing row group {row_group_idx + 1}/{num_row_groups} with {len(row_group_df)} rows")
                
                # Process the row group in batches
                for start_idx in range(0, len(row_group_df), batch_size):
                    end_idx = min(start_idx + batch_size, len(row_group_df))
                    batch_df = row_group_df.iloc[start_idx:end_idx]
                    
                    # Prepare batch lists
                    game_fen_blocks_batch = []      # List of tensors, each (1, *fen_shape)
                    game_move_sequences_batch = []  # List of tensors, each (1,) with padding move
                    raw_fens_for_output_batch = []  # List of lists of strings, if triple or return_fen
                    value_batch = []                # List of outcome indices
                    value_q_batch = []              # List of q values
                    # Process each row in the batch
                    for _, row in batch_df.iterrows():
                        fen, wdl,q = row.iloc[0], row.iloc[3], row.iloc[1]
                        white_win, draw_prob, black_win = wdl[0], wdl[1], wdl[2]
                        q_win,q_draw,q_loss = q[0],q[1],q[2]
                        # Apply mirroring with 50% probability for data augmentation
                        if random.random() < 0.5:
                            fen= mirror_fen(fen)
                            white_win, draw_prob, black_win = black_win, draw_prob, white_win
                            q_win,q_draw,q_loss = q_loss,q_draw,q_win
                        # Process the single FEN
                        current_fen_str = fen
                        current_game_fens_encoded_list = [torch.from_numpy(fen_to_tensor(current_fen_str))]
                        
                        # Stack the game's FEN into a single block tensor (1, *fen_shape)
                        current_game_fen_block = torch.stack(current_game_fens_encoded_list)
                        game_fen_blocks_batch.append(current_game_fen_block)
                        
                        # Add raw FEN if needed
                        if return_fen or triple:
                            raw_fens_for_output_batch.append([current_fen_str])
                        
                        # Always use padding move (single move per position)
                        padding_move = get_padding_move()
                        encoded_game_moves = torch.tensor([padding_move], dtype=torch.long)
                        game_move_sequences_batch.append(encoded_game_moves)
                        
                        # Process probabilities - get the index of the winning outcome
                        wdl,q = process_probabilities(fen, white_win, draw_prob, black_win,q_win,q_draw,q_loss)
                        #print(fen,outcome_index)
                        value_batch.append(wdl)
                        value_q_batch.append(q)

                    # Yield the batch
                    yield from _yield_batch(
                        game_fen_blocks_batch,
                        game_move_sequences_batch,
                        raw_fens_for_output_batch if (return_fen or triple) else None,
                        return_fen,
                        triple,
                        device,
                        value_batch,
                        value_q_batch,
                        block_size
                    )
                    
            except Exception as e:
                print(f"Error processing row group {row_group_idx} from {file_path}: {e}")
                continue


def _yield_batch(
        fen_blocks_list,         # List of [1, *fen_shape] tensors
        move_sequences_list,     # List of [1] move index tensors (padding)
        raw_fens_list_of_lists,  # List of lists of FEN strings (each inner list has 1 FEN), or None
        return_fen_flag,
        triple_flag,
        device,
        value_batch,             # List of outcome indices
        value_q_batch,
        block_size
    ):
    if return_fen_flag:
        # raw_fens_list_of_lists is expected to be a list where each item is 
        # a list of 1 FEN string
        yield raw_fens_list_of_lists
    else:
        fens_batch_tensor = torch.stack(fen_blocks_list).to(device)  # (batch_size, 1, 8, 8, 19)
        moves_batch_tensor = torch.stack(move_sequences_list).to(device)  # (batch_size, 1)
        
        # Stack value arrays to get (batch_size, 3)
        value_numpy = np.array(value_batch).reshape(-1, 3)  # (batch_size, 3)
        value_batch_tensor = torch.from_numpy(value_numpy).to(device)
        value_q_numpy = np.array(value_q_batch).reshape(-1, 3)  # (batch_size, 1)
        value_q_tensor = torch.from_numpy(value_q_numpy).to(device)

        if triple_flag:
            # raw_fens_list_of_lists is the batch of FEN strings
            yield fens_batch_tensor, moves_batch_tensor, raw_fens_list_of_lists, value_batch_tensor,value_q_tensor
        else:
            yield fens_batch_tensor, moves_batch_tensor, value_batch_tensor,value_q_tensor


if __name__ == '__main__':
    from tqdm import tqdm

    # Test triple format
    print("\nTesting triple=True:")
    try:
        gen_triple = batch_generator(input_dir, batch_size=32, block_size=1, triple=True, device=('cuda' if torch.cuda.is_available() else 'cpu'))
        for i in tqdm(range(10)):
            f_batch_t, m_batch_t, raw_fen_batch_t, value_batch_t,value_q_batch_t = next(gen_triple)
            print(f"Batch {i+1} (Triple):")
            print(f"  FENs tensor shape: {f_batch_t.shape}")
            print(f"  Moves tensor shape: {m_batch_t.shape}")
            print(f"  Value tensor shape: {value_batch_t.shape}")
            print(f"  Value Q tensor shape: {value_q_batch_t.shape}")
            if isinstance(raw_fen_batch_t, list) and len(raw_fen_batch_t) > 0:
                print(f"  Number of positions in raw FENs batch: {len(raw_fen_batch_t)}")
                print(f"  First FEN string: {raw_fen_batch_t[0][0]}")
                print(f"  Sample values from first position: {value_batch_t[0]}, {value_q_batch_t[0]}")
            else:
                print(f"  Received unexpected data format for raw FENs in triple: {type(raw_fen_batch_t)}")
    except Exception as e:
        print(f"Error during triple test: {e}")

    print("\nTests complete.")
