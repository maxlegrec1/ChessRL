import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from collections import deque
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

input_dir = "/mnt/2tb/LeelaDataReader/output_positions3"

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


def mirror_fen_and_probabilities(fen, win_prob, draw_prob, loss_prob):
    """
    Mirror the FEN to change perspective from white to black and vice versa.
    Also swap win_prob and loss_prob since the perspective changes.
    
    Args:
        fen: Original FEN string
        win_prob: Probability that the side to move wins
        draw_prob: Probability of draw  
        loss_prob: Probability that the side to move loses
        
    Returns:
        tuple: (mirrored_fen, mirrored_win_prob, draw_prob, mirrored_loss_prob)
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
        return fen, win_prob, draw_prob, loss_prob
    
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
    
    # Swap win_prob and loss_prob since we've changed perspective
    mirrored_win_prob = loss_prob
    mirrored_loss_prob = win_prob
    # draw_prob stays the same
    
    return mirrored_fen, mirrored_win_prob, draw_prob, mirrored_loss_prob


def process_probabilities(fen, win_prob, draw_prob, loss_prob):
    """
    Convert probabilities to winning side index.
    
    Args:
        fen: FEN string to determine whose turn it is
        win_prob: Probability that the side to move wins
        draw_prob: Probability of draw
        loss_prob: Probability that the side to move loses
        
    Returns:
        int: Index of the outcome (0=black wins, 1=draw, 2=white wins)
    """
    # Determine whose turn it is from the FEN
    # FEN format: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # The second-to-last field before the move counters indicates whose turn: 'w' or 'b'
    fen_parts = fen.split()
    if len(fen_parts) < 2:
        # Malformed FEN, assume white to move
        active_color = 'w'
    else:
        active_color = fen_parts[1]
    
    if active_color == 'b':
        # Black to move: win_prob = black wins, loss_prob = white wins
        black_win_prob = win_prob
        white_win_prob = loss_prob
    else:
        # White to move: win_prob = white wins, loss_prob = black wins
        black_win_prob = loss_prob
        white_win_prob = win_prob
    
    # Find the index of the highest probability
    probs = [black_win_prob, draw_prob, white_win_prob]
    return np.argmax(probs)


def batch_generator(input_dir, batch_size, block_size=1, return_fen=False, triple=False, device='cuda', buffer_size=6_000_000, refill_frequency=10):
    """
    Yield batches of encoded FENs and padding moves from parquet files using an optimized buffer system.
    
    The buffer maintains a constant mix of data from different files to avoid training
    on clustered data from the same games. Buffer is refilled in chunks every N batches
    for better performance.
    
    Args:
        input_dir: Directory containing parquet files
        batch_size: Number of positions per batch
        block_size: Always 1 for this version
        return_fen: Whether to return raw FEN strings instead of tensors
        triple: Whether to return (tensors, moves, raw_fens, values)
        device: Device to place tensors on
        buffer_size: Size of the data buffer (default 6M to match typical file size)
        refill_frequency: Refill buffer every N batches (default 10)
    """
    if _zero_fen_pad_tensor is None:
        raise RuntimeError("FEN padding tensor initialization failed. Cannot proceed.")

    # Force block_size to be 1
    block_size = 1

    files = sorted([os.path.join(input_dir, f)
                    for f in os.listdir(input_dir) if f.endswith('.parquet')])
    # Shuffle files 
    random.shuffle(files)
    
    # Initialize buffer and file tracking - using list for O(1) swap-and-pop
    buffer = []
    current_file_idx = 0
    current_df = None
    current_row_idx = 0
    batch_count = 0
    
    def load_next_file():
        """Load the next file and return its dataframe, or None if no more files."""
        nonlocal current_file_idx, current_df, current_row_idx
        if current_file_idx >= len(files):
            return None
        
        print(f"Loading file {current_file_idx + 1}/{len(files)}: {files[current_file_idx]}")
        df = pd.read_parquet(files[current_file_idx])
        # Shuffle rows within the file
        df = df.sample(frac=1.0).reset_index(drop=True)
        current_file_idx += 1
        current_row_idx = 0
        return df
    
    def refill_buffer(target_refill_size):
        """Refill buffer with target_refill_size rows by loading data from current/next files."""
        nonlocal current_df, current_row_idx
        
        refill_count = 0
        rows_to_add = []
        
        while refill_count < target_refill_size:
            # If no current dataframe, load the next file
            if current_df is None:
                current_df = load_next_file()
                if current_df is None:
                    # No more files to load
                    break
            
            # Calculate how many rows to take from current file
            rows_needed = target_refill_size - refill_count
            rows_available = len(current_df) - current_row_idx
            rows_to_take = min(rows_needed, rows_available)
            
            # Extract rows in batch using iloc for efficiency
            if rows_to_take > 0:
                batch_rows = current_df.iloc[current_row_idx:current_row_idx + rows_to_take]
                # Convert to list of tuples more efficiently
                for _, row in batch_rows.iterrows():
                    rows_to_add.append((row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]))
                
                current_row_idx += rows_to_take
                refill_count += rows_to_take
            
            # If we've exhausted the current file, mark it for reloading
            if current_row_idx >= len(current_df):
                current_df = None
        
        # Add all new rows to buffer at once
        buffer.extend(rows_to_add)
        return refill_count
    
    # Initial buffer fill
    print(f"Initializing buffer with {buffer_size:,} rows...")
    initial_fill = refill_buffer(buffer_size)
    print(f"Buffer initialized with {len(buffer):,} rows")
    
    # Pre-allocate arrays for efficient sampling
    sample_indices = np.empty(batch_size, dtype=np.int32)
    
    # Batch generation
    game_fen_blocks_batch = []      # List of tensors, each (1, *fen_shape)
    game_move_sequences_batch = []  # List of tensors, each (1,) with padding move
    raw_fens_for_output_batch = []  # List of lists of strings, if triple or return_fen
    value_batch = []                # List of outcome indices

    while buffer:
        # Check if we need to refill buffer (every refill_frequency batches)
        if batch_count % refill_frequency == 0 and batch_count > 0:
            # Calculate refill size (replace consumed elements)
            refill_size = min(refill_frequency * batch_size, buffer_size - len(buffer))
            if refill_size > 0 and (current_df is not None or current_file_idx < len(files)):
                refilled = refill_buffer(refill_size)
                if refilled > 0:
                    print(f"Refilled buffer with {refilled:,} rows, buffer size: {len(buffer):,}")
        
        # Sample batch_size indices efficiently
        if len(buffer) < batch_size:
            # Take all remaining if buffer is smaller than batch_size
            batch_size_actual = len(buffer)
            sample_indices_actual = np.arange(batch_size_actual)
        else:
            batch_size_actual = batch_size
            sample_indices_actual = np.random.choice(len(buffer), size=batch_size_actual, replace=False)
        
        # Sort indices in descending order for efficient removal (remove from end first)
        sample_indices_sorted = np.sort(sample_indices_actual)[::-1]
        
        # Extract sampled rows and remove from buffer (from end to beginning to maintain indices)
        sampled_rows = []
        for idx in sample_indices_sorted:
            sampled_rows.append(buffer[idx])
            # Efficient removal: swap with last element and pop
            if idx < len(buffer) - 1:
                buffer[idx] = buffer[-1]
            buffer.pop()
        
        # Reverse to restore original sampling order
        sampled_rows.reverse()
        
        # Process the sampled batch
        for fen, win_prob, draw_prob, loss_prob in sampled_rows:
            # Apply mirroring with 50% probability for data augmentation
            if random.random() < 0.5:
                fen, win_prob, draw_prob, loss_prob = mirror_fen_and_probabilities(fen, win_prob, draw_prob, loss_prob)
            
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
            outcome_index = process_probabilities(fen, win_prob, draw_prob, loss_prob)
            value_batch.append(outcome_index)
        
        # Yield the batch
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
        
        # Clear batch lists
        game_fen_blocks_batch.clear()
        game_move_sequences_batch.clear()
        if return_fen or triple:
            raw_fens_for_output_batch.clear()
        value_batch.clear()
        
        batch_count += 1


def _yield_batch(
        fen_blocks_list,         # List of [1, *fen_shape] tensors
        move_sequences_list,     # List of [1] move index tensors (padding)
        raw_fens_list_of_lists,  # List of lists of FEN strings (each inner list has 1 FEN), or None
        return_fen_flag,
        triple_flag,
        device,
        value_batch,             # List of outcome indices
        block_size
    ):
    if return_fen_flag:
        # raw_fens_list_of_lists is expected to be a list where each item is 
        # a list of 1 FEN string
        yield raw_fens_list_of_lists
    else:
        fens_batch_tensor = torch.stack(fen_blocks_list).to(device)  # (batch_size, 1, 8, 8, 19)
        moves_batch_tensor = torch.stack(move_sequences_list).to(device)  # (batch_size, 1)
        
        # Stack value arrays to get (batch_size, 1)
        value_numpy = np.array(value_batch).reshape(-1, 1)  # (batch_size, 1)
        value_batch_tensor = torch.from_numpy(value_numpy).to(device)
        
        if triple_flag:
            # raw_fens_list_of_lists is the batch of FEN strings
            yield fens_batch_tensor, moves_batch_tensor, raw_fens_list_of_lists, value_batch_tensor
        else:
            yield fens_batch_tensor, moves_batch_tensor, value_batch_tensor


if __name__ == '__main__':
    from tqdm import tqdm

    # Test with the new format
    print("\nTesting new probability-based format:")
    try:
        gen = batch_generator(input_dir, batch_size=32, block_size=1, device=('cuda' if torch.cuda.is_available() else 'cpu'))
        for i in tqdm(range(2)):
            f_batch_t, m_batch_t, value_batch_t = next(gen)
            print(f"Batch {i+1}:")
            print(f"  FENs tensor shape: {f_batch_t.shape}")  # Should be (32, 1, 8, 8, 19)
            print(f"  Moves tensor shape: {m_batch_t.shape}")  # Should be (32, 1)
            print(f"  Value tensor shape: {value_batch_t.shape}")  # Should be (32, 1)
            print(f"  Sample values from first position: {value_batch_t[0]}")  # Outcome index
            print(f"  All moves are padding (should be {_pad_id}): {torch.all(m_batch_t == _pad_id)}")
    except Exception as e:
        print(f"Error during test: {e}")

    # Test triple format
    print("\nTesting triple=True:")
    try:
        gen_triple = batch_generator(input_dir, batch_size=32, block_size=1, triple=True, device=('cuda' if torch.cuda.is_available() else 'cpu'))
        for i in tqdm(range(2)):
            f_batch_t, m_batch_t, raw_fen_batch_t, value_batch_t = next(gen_triple)
            print(f"Batch {i+1} (Triple):")
            print(f"  FENs tensor shape: {f_batch_t.shape}")
            print(f"  Moves tensor shape: {m_batch_t.shape}")
            print(f"  Value tensor shape: {value_batch_t.shape}")
            if isinstance(raw_fen_batch_t, list) and len(raw_fen_batch_t) > 0:
                print(f"  Number of positions in raw FENs batch: {len(raw_fen_batch_t)}")
                print(f"  First FEN string: {raw_fen_batch_t[0][0]}")
            else:
                print(f"  Received unexpected data format for raw FENs in triple: {type(raw_fen_batch_t)}")
    except Exception as e:
        print(f"Error during triple test: {e}")

    print("\nTests complete.")
