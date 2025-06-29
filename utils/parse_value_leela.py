import os
import sys
import yaml
import numpy as np
import torch
import random
try:
    from utils.fen_encoder import fen_to_tensor
except ImportError:
    from fen_encoder import fen_to_tensor

#get data_generator from /mnt/2tb/LeelaDataReader/lczero-training/tf/inspect_data_native.py
#it has dependencies in /mnt/2tb/LeelaDataReader/lczero-training/tf/
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
leela_tf_path = '/mnt/2tb/LeelaDataReader/lczero-training/tf'
if leela_tf_path not in sys.path:
    sys.path.append(leela_tf_path)



try:
    from chunkparser import ChunkParser
    from inspect_data_native import get_input_mode, get_latest_chunks, identity_function, planes_to_fen
except ImportError as e:
    print(f"Failed to import from {leela_tf_path}: {e}")
    sys.exit(1)


def leela_data_generator(config_path):
    """
    A generator that yields batches of data from Leela's chunk files,
    providing FEN strings, Q-values, and winner probabilities.

    Args:
        config_path (str): Path to the YAML configuration file for Leela data.

    Yields:
        tuple: A tuple containing (fens, q_values, winner_probs).
               - fens (list): A list of FEN strings for the batch.
               - q_values (np.array): Numpy array of Q-values.
               - winner_probs (np.array): Numpy array of winner probabilities.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    num_chunks = cfg["dataset"]["num_chunks"]
    allow_less = cfg["dataset"].get("allow_less_chunks", False)
    sort_key_fn = identity_function
    fast_chunk_loading = cfg["dataset"].get("fast_chunk_loading", True)

    if "input_train" in cfg["dataset"]:
        chunks = get_latest_chunks(cfg["dataset"]["input_train"],
                                         num_chunks, allow_less, sort_key_fn, fast=fast_chunk_loading)
    elif "input" in cfg["dataset"]:
         chunks = get_latest_chunks(cfg["dataset"]["input"],
                                     num_chunks, allow_less, sort_key_fn, fast=fast_chunk_loading)
    else:
        raise ValueError("Config must contain 'input_train' or 'input' key in dataset.")

    shuffle_size = cfg["training"]["shuffle_size"]
    batch_size = cfg["training"]["batch_size"]

    parser = ChunkParser(chunks,
                         get_input_mode(cfg),
                         shuffle_size=shuffle_size,
                         batch_size=batch_size)

    for batch in parser.parse():
        planes_buf, probs_buf, winner_buf, q_buf, _, _, _, _, _ = batch

        planes = np.frombuffer(planes_buf, dtype=np.float32).reshape(
            (batch_size, 112, 8, 8))
        #permute to get (B,8,8,112
        fens = planes_to_fen(planes)
        planes = planes.transpose(0,2,3,1)
        probs_buf = np.frombuffer(probs_buf, dtype=np.float32).reshape(
            (batch_size, 1858))
        winner = np.frombuffer(winner_buf, dtype=np.float32).reshape(
            (batch_size, 3)).copy()
        q = np.frombuffer(q_buf, dtype=np.float32).reshape(
            (batch_size, 3)).copy()
        #mirror fens randomly when do, we also have to mirror value and q # and ultimately move
        # Mirror FENs randomly for data augmentation
        # for i in range(batch_size):
        #     if random.random() < 0.5:
        #         fens[i] = mirror_fen(fens[i])
                
                
        #     else:
        #         # Mirror the winner probabilities (swap white and black)
        #         winner[i] = winner[i][::-1]  # Reverse the array to swap white/black
        #         # Mirror the Q-values (swap white and black)
        #         q[i] = q[i][::-1]  # Reverse the array to swap white/black

        #has to yield : pos,move,fen,value,value
        #pos = torch.from_numpy(np.array([fen_to_tensor(fens[i]) for i in range(batch_size)])).to("cuda")
        pos = torch.from_numpy(planes).to("cuda")
        pos = pos.unsqueeze(1)
        move = torch.from_numpy(np.argmax(probs_buf, axis=1)).to("cuda")
        value = torch.from_numpy(winner).to("cuda")
        q = torch.from_numpy(q).to("cuda")
        yield pos, move,fens, value, q


if __name__ == '__main__':
    # An example of how to use the generator.
    # A dummy config is created for demonstration.
    # The user should replace "leela_config.yaml" with a valid config file.
    config_path = "/mnt/2tb/LeelaDataReader/lczero-training/tf/configs/example.yaml"
    print(f"Running example with config: {config_path}")
    print("This will likely fail if the data path in the config is not valid.")
    
    try:
        gen = leela_data_generator(config_path)
        for i in range(10):
            pos, move, fens, value, q = next(gen)
            print(pos.shape)
            print(f"\n--- Batch {i+1} ---")
            print(f"  Number of positions: {len(fens)}")
            print(f"  Q-values shape: {q.shape}")
            print(f"  Winner probs shape: {value.shape}")
            print(f"  First FEN: {fens[0]}")
            print(f" First Q: {q[0]}")
            print(f" First Value: {value[0]}")
    except Exception as e:
        print(f"\nAn error occurred during generator execution: {e}")
        print("Please ensure the path in your config file points to valid Leela chunk files.")

