#!/usr/bin/env python3
"""
Load random batches from training data parquet file.

Reads batch_size rows randomly and creates tensors:
- X: batch of board encodings (positions)
- policy: batch of policy indexes
- R: batch of side_to_move_won
- wdl: batch of wdl indexes
"""

import argparse
import sys
import os

import numpy as np
import pyarrow.parquet as pq
import torch


def load_random_batch(parquet_file_path: str, batch_size: int, verbose: bool = True) -> tuple:
    """
    Load a random batch of rows from the parquet file.
    
    Memory-efficient: reads the file in batches and samples from them.
    
    Args:
        parquet_file_path: Path to the parquet file
        batch_size: Number of rows to sample
        verbose: Whether to print progress logs
    
    Returns:
        Tuple of (X, policy, R, wdl) tensors:
        - X: Tensor of shape (batch_size, 1216) - board encodings
        - policy: Tensor of shape (batch_size,) - policy indexes
        - R: Tensor of shape (batch_size,) - side_to_move_won
        - wdl: Tensor of shape (batch_size,) - wdl indexes
    """
    if verbose:
        print(f"  [DEBUG] Opening parquet file: {parquet_file_path}")
    
    # Open parquet file
    parquet_file = pq.ParquetFile(parquet_file_path)
    total_rows = parquet_file.metadata.num_rows
    
    if verbose:
        print(f"  [DEBUG] Total rows in file: {total_rows}")
    
    # Sample random row indices and sort them for efficient reading
    if batch_size > total_rows:
        batch_size = total_rows
    random_indices = sorted(np.random.choice(total_rows, size=batch_size, replace=False))
    random_indices_set = set(random_indices)  # For fast lookup
    
    if verbose:
        print(f"  [DEBUG] Selected {len(random_indices)} random indices")
        print(f"  [DEBUG] Indices range: {random_indices[0]} to {random_indices[-1]}")
        print(f"  [DEBUG] Will read up to row {random_indices[-1]} (out of {total_rows} total)")
    
    # Collect all rows first (we need random access)
    # Memory-efficient: read in batches and only keep rows we need
    collected_rows = []
    current_row_idx = 0
    batch_read_size = 1000  # Read 1000 rows at a time
    batch_num = 0
    next_target_idx = 0  # Index into sorted random_indices
    
    if verbose:
        print(f"  [DEBUG] Starting to read batches (batch_read_size={batch_read_size})...")
    
    for batch in parquet_file.iter_batches(batch_size=batch_read_size):
        batch_num += 1
        
        # Skip batches that are before our first target index
        batch_end = current_row_idx + batch_read_size
        if batch_end <= random_indices[0]:
            current_row_idx = batch_end
            if verbose and batch_num % 100 == 0:
                print(f"  [DEBUG] Skipping batch {batch_num} (row {current_row_idx} < target {random_indices[0]})...")
            continue
        
        if verbose and (batch_num % 10 == 0 or next_target_idx < len(random_indices)):
            print(f"  [DEBUG] Reading batch {batch_num} (row {current_row_idx} to {batch_end})...")
        
        batch_df = batch.to_pandas()
        
        # Check which rows in this batch we need
        rows_found = 0
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            # Check if we've passed all targets
            if next_target_idx >= len(random_indices):
                break
                
            if current_row_idx == random_indices[next_target_idx]:
                collected_rows.append({
                    'board_encoding': np.array(row['board_encoding'], dtype=np.float32),
                    'policy_index': int(row['policy_index']),
                    'side_to_move_won': int(row['side_to_move_won']),
                    'wdl': int(row['wdl']),
                })
                rows_found += 1
                next_target_idx += 1
                if verbose:
                    print(f"  [DEBUG] Found row {current_row_idx}, collected {rows_found}/{batch_size} rows")
            
            current_row_idx += 1
        
        # Clear batch from memory
        del batch_df
        del batch
        
        if verbose and rows_found > 0:
            print(f"  [DEBUG] Batch {batch_num} processed. Collected {len(collected_rows)}/{batch_size} rows so far")
        
        # Stop if we've collected enough
        if len(collected_rows) >= batch_size:
            break
        
        # Stop if we've passed all target indices
        if current_row_idx > random_indices[-1]:
            break
    
    if verbose:
        print(f"  [DEBUG] Finished reading. Collected {len(collected_rows)} rows")
        if len(collected_rows) < batch_size:
            print(f"  [WARNING] Only collected {len(collected_rows)} rows but expected {batch_size}")
        print(f"  [DEBUG] Converting to tensors...")
    
    # Convert to tensors
    board_encodings = [row['board_encoding'] for row in collected_rows]
    policy_indexes = [row['policy_index'] for row in collected_rows]
    side_to_move_won = [row['side_to_move_won'] for row in collected_rows]
    wdl_values = [row['wdl'] for row in collected_rows]
    
    if verbose:
        print(f"  [DEBUG] Creating X tensor from {len(board_encodings)} encodings...")
    # Reshape from (batch_size, 1216) to (batch_size, 8, 8, 19)
    # The encoding was flattened when stored in parquet, but fen_to_tensor returns (8, 8, 19)
    X = torch.from_numpy(np.array(board_encodings)).reshape(len(board_encodings), 8, 8, 19)  # (batch_size, 8, 8, 19)
    
    if verbose:
        print(f"  [DEBUG] Creating policy tensor...")
    policy = torch.tensor(policy_indexes, dtype=torch.long)  # (batch_size,)
    
    if verbose:
        print(f"  [DEBUG] Creating R tensor...")
    R = torch.tensor(side_to_move_won, dtype=torch.float32)  # (batch_size,)
    
    if verbose:
        print(f"  [DEBUG] Creating wdl tensor...")
    wdl = torch.tensor(wdl_values, dtype=torch.long)  # (batch_size,)
    
    if verbose:
        print(f"  [DEBUG] All tensors created successfully")
    
    return X, policy, R, wdl


def main():
    parser = argparse.ArgumentParser(
        description='Load random batches from training data'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/training_data.parquet',
        help='Input parquet file path (default: data/training_data.parquet)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading batches from: {args.input}")
    print(f"Batch size: {args.batch_size}\n")
    
    # Load and display 5 batches
    for batch_idx in range(5):
        print(f"\nBatch {batch_idx + 1}:")
        print("=" * 60)
        try:
            X, policy, R, wdl = load_random_batch(args.input, args.batch_size, verbose=True)
            
            print(f"  X shape: {X.shape}")
            print(f"  policy shape: {policy.shape}")
            print(f"  R shape: {R.shape}")
            print(f"  wdl shape: {wdl.shape}")
        except Exception as e:
            print(f"  ERROR: Failed to load batch: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == '__main__':
    main()

