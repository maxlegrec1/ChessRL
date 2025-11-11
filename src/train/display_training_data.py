#!/usr/bin/env python3
"""
Read and display training data from parquet file.

Shows row ID, FEN, encoded position shape, UCI move, policy index,
side_to_move_won, and wdl for each row.

Memory-efficient: reads rows one at a time without loading entire file.
"""

import argparse
import sys

import pyarrow.parquet as pq
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Display training data from parquet file'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/training_data.parquet',
        help='Input parquet file path (default: data/training_data.parquet)'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum number of rows to display (default: all rows)'
    )
    
    args = parser.parse_args()
    
    # Open parquet file for reading
    try:
        parquet_file = pq.ParquetFile(args.input)
    except Exception as e:
        print(f"Error: Failed to open parquet file {args.input}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get total number of rows from metadata (very memory efficient)
    total_rows = parquet_file.metadata.num_rows
    
    # Determine number of rows to display
    num_rows_to_display = total_rows
    if args.max_rows is not None:
        num_rows_to_display = min(total_rows, args.max_rows)
    
    print(f"Total rows in file: {total_rows}")
    print(f"Displaying first {num_rows_to_display} rows\n")
    print("=" * 100)
    
    # Read rows in batches (memory-efficient)
    row_idx = 0
    batch_size = 100  # Read 100 rows at a time
    
    # Iterate through row groups
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        # Convert batch to pandas for easier access (only 100 rows at a time)
        batch_df = batch.to_pandas()
        
        # Display rows in this batch
        for _, row in batch_df.iterrows():
            if row_idx >= num_rows_to_display:
                return
            
            # Get board encoding as numpy array to check shape
            board_encoding = np.array(row['board_encoding'])
            board_encoding = board_encoding.reshape(8, 8, 19)
            print(f"\nRow ID: {row_idx}")
            print(f"FEN: {row['fen']}")
            print(f"Encoded position shape: {board_encoding.shape}")
            print(f"UCI move: {row['uci_move']}")
            print(f"Policy index: {row['policy_index']}")
            print(f"side_to_move_won: {row['side_to_move_won']}")
            print(f"wdl: {row['wdl']}")
            print("-" * 100)
            
            row_idx += 1
        
        # Clear batch from memory before next iteration
        del batch_df
        del batch


if __name__ == '__main__':
    main()

