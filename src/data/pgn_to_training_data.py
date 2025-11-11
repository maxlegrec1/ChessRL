#!/usr/bin/env python3
"""
Transform PGN files to training data for the small model.

Processes PGN files from runs/pgn/ and extracts positions after opening moves.
Memory-efficient implementation that processes games sequentially and writes
parquet files in chunks.
"""

import argparse
import os
import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import bulletchess
from bulletchess import WHITE, BLACK
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


from src.models.small.fen_encoder import fen_to_tensor
from src.models.small.vocab import policy_to_idx


def parse_result(result_str: str) -> Tuple[int, int]:
    """
    Parse PGN result string to side_to_move_won and wdl.
    
    Args:
        result_str: PGN result string ("1-0", "0-1", "1/2-1/2", or "*")
    
    Returns:
        Tuple of (side_to_move_won, wdl):
        - side_to_move_won: 1 if current side won, -1 if lost, 0 if draw
        - wdl: 0 if BLACK won, 1 if draw, 2 if WHITE won
    """
    if result_str == "1-0":
        # White won
        return (1, 2)  # (side_to_move_won will be set based on turn), (wdl=2 for white)
    elif result_str == "0-1":
        # Black won
        return (-1, 0)  # (side_to_move_won will be set based on turn), (wdl=0 for black)
    elif result_str == "1/2-1/2":
        # Draw
        return (0, 1)  # (side_to_move_won=0 for draw), (wdl=1 for draw)
    else:
        # Unknown result
        return (0, 1)  # Default to draw


def get_move_uci_and_index(move: bulletchess.Move) -> Tuple[str, int]:
    """
    Extract UCI move string and policy index.
    
    Args:
        move: bulletchess Move object
    
    Returns:
        Tuple of (uci_move, policy_index)
    """
    uci_move = move.uci()
    
    # Handle castling moves (remove 'n' suffix if present)
    if uci_move.endswith('n'):
        uci_move = uci_move[:-1]
    
    # Get policy index
    policy_index = policy_to_idx.get(uci_move, -1)
    
    return uci_move, policy_index


def process_game(
    game,
    result_str: str,
    opening_length,
    players: list
) -> List[Dict]:
    """
    Process a single game and extract positions after opening.
    
    Args:
        game: PGNGame object from bulletchess.pgn
        result_str: Game result string
    
    Returns:
        List of position dictionaries with training data
    """
    positions = []
    
    # Parse result
    _, wdl = parse_result(result_str)
    
    # Get starting board (after opening moves)
    board = game.starting_board.copy()
    
    # Process each move after opening
    for i,move in enumerate(game.moves):
        if i < opening_length or board.turn not in players:
            board.apply(move)
            continue
        # Get current position info before applying move
        fen = board.fen()
        is_white_turn = board.turn == bulletchess.WHITE
        
        # Get board encoding
        try:
            board_tensor = fen_to_tensor(fen)
            # Flatten to 1D array (8*8*19 = 1216)
            board_encoding = board_tensor.flatten().astype(np.int16)
        except Exception as e:
            print(f"Warning: Failed to encode FEN {fen}: {e}", file=sys.stderr)
            board.apply(move)
            continue
        
        # Get move info
        uci_move, policy_idx = get_move_uci_and_index(move)
        
        # Calculate side_to_move_won
        if result_str == "1-0":
            side_to_move_won = 1 if is_white_turn else -1
        elif result_str == "0-1":
            side_to_move_won = -1 if is_white_turn else 1
        elif result_str == "1/2-1/2":
            side_to_move_won = 0
        else:
            side_to_move_won = 0  # Unknown result
        
        # Store position data
        positions.append({
            'fen': fen,
            'board_encoding': board_encoding,
            'uci_move': uci_move,
            'policy_index': policy_idx,
            'side_to_move_won': np.int8(side_to_move_won),
            'wdl': np.int8(wdl),
        })
        
        # Apply move to board
        board.apply(move)
    
    return positions


def create_parquet_schema() -> pa.Schema:
    """Create PyArrow schema for the output parquet file."""
    return pa.schema([
        ('fen', pa.string()),
        ('board_encoding', pa.list_(pa.int16(), 1216)),  # 8*8*19 = 1216
        ('uci_move', pa.string()),
        ('policy_index', pa.int32()),
        ('side_to_move_won', pa.int8()),
        ('wdl', pa.int8()),
    ])


def write_batch_to_parquet(
    writer: pq.ParquetWriter,
    batch: List[Dict],
    schema: pa.Schema
) -> None:
    """
    Write a batch of positions to parquet file.
    
    Args:
        writer: ParquetWriter instance
        batch: List of position dictionaries
        schema: PyArrow schema
    """
    if not batch:
        return
    
    # Convert to PyArrow arrays
    arrays = {
        'fen': [pos['fen'] for pos in batch],
        'board_encoding': [pos['board_encoding'].tolist() for pos in batch],
        'uci_move': [pos['uci_move'] for pos in batch],
        'policy_index': [pos['policy_index'] for pos in batch],
        'side_to_move_won': [pos['side_to_move_won'] for pos in batch],
        'wdl': [pos['wdl'] for pos in batch],
    }
    
    # Create table and write
    table = pa.Table.from_pydict(arrays, schema=schema)
    writer.write_table(table)


def main():
    parser = argparse.ArgumentParser(
        description='Transform PGN files to training data'
    )
    parser.add_argument(
        '--max-games',
        type=int,
        default=1,
        help='Maximum number of games to process (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='src/data/training_data.parquet',
        help='Output parquet file path (default: src/data/training_data.parquet)'
    )
    parser.add_argument(
        '--pgn-dir',
        type=str,
        default='runs/pgn',
        help='Directory containing PGN files (default: runs/pgn)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Number of positions to accumulate before writing (default: 10000)'
    )
    parser.add_argument(
        '--player_name',
        type=str,
        default=None,
        help='If not None, only record the move played by this player'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Get list of PGN files
    pgn_dir = args.pgn_dir
    if not os.path.exists(pgn_dir):
        print(f"Error: PGN directory not found: {pgn_dir}", file=sys.stderr)
        sys.exit(1)
    
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    pgn_files.sort()  # Process in consistent order
    
    if not pgn_files:
        print(f"Warning: No PGN files found in {pgn_dir}", file=sys.stderr)
        return
    
    print(f"Found {len(pgn_files)} PGN files")
    print(f"Processing up to {args.max_games} games")
    print(f"Output file: {args.output}")
    
    # Initialize parquet writer
    schema = create_parquet_schema()
    writer = pq.ParquetWriter(args.output, schema)
    
    # Process games
    games_processed = 0
    total_positions = 0
    batch = []
    
    try:
        with tqdm(total=args.max_games, desc="Processing games") as pbar:
            for pgn_file in pgn_files:
                if games_processed >= args.max_games:
                    break
                
                pgn_path = os.path.join(pgn_dir, pgn_file)
                
                try:
                    with bulletchess.pgn.PGNFile.open(pgn_path) as f:
                        game = f.next_game()
                        if game is None:
                            continue
                        
                        # Extract headers (use dictionary access like bulletchess_pgnreader.py)

                        opening_uci_str = game["OpeningUCI"]
                        result_str = game["Result"]

                        if args.player_name is None:
                            players = [WHITE,BLACK]
                        else:
                            if game["White"] == args.player_name:
                                players = [WHITE]
                            elif game["Black"] == args.player_name:
                                players = [BLACK]
                            else:
                                print(f" WARNING : Neither player of this game is {args.player_name}. Players are {game["White"]} and {game["Black"]}")
                                players = [WHITE,BLACK]
                        print(f" Player {args.player_name} is {players}")
                        
                        # Parse opening moves
                        opening_moves = opening_uci_str.split() if opening_uci_str else []
                        len_opening = len(opening_moves)
                        # Skip games with unknown result
                        if result_str not in ["1-0", "0-1", "1/2-1/2"]:
                            continue
                        
                        # Process game
                        positions = process_game(game, result_str, len_opening, players)
                        
                        # Add to batch
                        batch.extend(positions)
                        
                        # Write batch if it's large enough
                        if len(batch) >= args.chunk_size:
                            write_batch_to_parquet(writer, batch, schema)
                            total_positions += len(batch)
                            batch = []  # Clear batch to free memory
                        
                        games_processed += 1
                        pbar.update(1)
                        pbar.set_postfix({"positions": total_positions + len(batch)})
                
                except Exception as e:
                    print(f"Warning: Failed to process {pgn_file}: {e}", file=sys.stderr)
                    continue
        
        # Write remaining batch
        if batch:
            write_batch_to_parquet(writer, batch, schema)
            total_positions += len(batch)
    
    finally:
        writer.close()
    
    print(f"\nCompleted!")
    print(f"Games processed: {games_processed}")
    print(f"Total positions extracted: {total_positions}")
    print(f"Output file: {args.output}")


if __name__ == '__main__':
    main()

