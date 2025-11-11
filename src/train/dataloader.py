#!/usr/bin/env python3
"""
PyTorch DataLoader for chess training data from parquet files.

Loads batches from shuffled parquet file and provides:
- X: board positions (N, 8, 8, 19)
- policy: UCI move indices (N,)
- R: side_to_move_won (N,)
- WDL: win/draw/loss labels (N,)
"""

import os
from typing import Tuple, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, DataLoader


class ChessDataset(IterableDataset):
    """
    Iterable dataset for chess training data stored in parquet format.
    
    Efficiently streams data from parquet file without loading everything into memory.
    """
    
    def __init__(
        self,
        parquet_file_path: str,
        batch_size: int = 32,
        shuffle: bool = False,
        chunk_size: int = 10000,
    ):
        """
        Initialize the dataset.
        
        Args:
            parquet_file_path: Path to the parquet file
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data within chunks (not globally)
            chunk_size: Number of rows to read at a time from parquet
        """
        super().__init__()
        
        if not os.path.exists(parquet_file_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")
        
        self.parquet_file_path = parquet_file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        
        # Get metadata
        parquet_file = pq.ParquetFile(parquet_file_path)
        self.total_rows = parquet_file.metadata.num_rows
        
    def __iter__(self):
        """
        Iterate over the dataset, yielding batches.
        
        Yields:
            Tuple of (X, policy, R, WDL) tensors for each batch
        """
        parquet_file = pq.ParquetFile(self.parquet_file_path)
        
        # Buffer for accumulating samples
        buffer = []
        
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            # Convert to pandas for easier processing
            batch_df = batch.to_pandas()
            
            # Convert to list of samples
            for _, row in batch_df.iterrows():
                sample = {
                    'board_encoding': np.array(row['board_encoding'], dtype=np.float32),
                    'policy_index': int(row['policy_index']),
                    'side_to_move_won': int(row['side_to_move_won']),
                    'wdl': int(row['wdl']),
                }
                buffer.append(sample)
                
                # Yield batch when buffer is full
                if len(buffer) >= self.batch_size:
                    yield self._create_batch(buffer[:self.batch_size])
                    buffer = buffer[self.batch_size:]
        
        # Yield remaining samples if any
        if len(buffer) > 0:
            yield self._create_batch(buffer)
    
    def _create_batch(self, samples: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a list of samples into a batch of tensors.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Tuple of (X, policy, R, WDL) tensors
        """
        if self.shuffle:
            np.random.shuffle(samples)
        
        # Extract fields
        board_encodings = np.array([s['board_encoding'] for s in samples], dtype=np.float32)
        policy_indexes = np.array([s['policy_index'] for s in samples], dtype=np.int64)
        side_to_move_won = np.array([s['side_to_move_won'] for s in samples], dtype=np.float32)
        wdl_values = np.array([s['wdl'] for s in samples], dtype=np.int64)
        
        # Create tensors
        # Reshape board encoding from (N, 1216) to (N, 8, 8, 19)
        X = torch.from_numpy(board_encodings).reshape(-1, 8, 8, 19)
        policy = torch.from_numpy(policy_indexes)
        R = torch.from_numpy(side_to_move_won)
        WDL = torch.from_numpy(wdl_values)
        
        return X, policy, R, WDL
    
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return self.total_rows


def create_dataloader(
    parquet_file_path: str = 'data/shuffled_data.parquet',
    batch_size: int = 32,
    shuffle: bool = False,
    chunk_size: int = 10000,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for chess training data.
    
    Args:
        parquet_file_path: Path to the parquet file
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data within chunks
        chunk_size: Number of rows to read at a time from parquet
        num_workers: Number of worker processes (0 = main process only)
        
    Returns:
        PyTorch DataLoader instance
    """
    dataset = ChessDataset(
        parquet_file_path=parquet_file_path,
        batch_size=batch_size,
        shuffle=shuffle,
        chunk_size=chunk_size,
    )
    
    # Note: For IterableDataset, the DataLoader batch_size should be None
    # because batching is handled by the dataset itself
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Batching handled by dataset
        num_workers=num_workers,
    )
    
    return dataloader


def main():
    """
    Demo usage of the dataloader.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test chess dataloader'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/shuffled_data.parquet',
        help='Input parquet file path (default: data/shuffled_data.parquet)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=5,
        help='Number of batches to display (default: 5)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading batches from: {args.input}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches to display: {args.num_batches}\n")
    
    # Create dataloader
    try:
        dataloader = create_dataloader(
            parquet_file_path=args.input,
            batch_size=args.batch_size,
            shuffle=True,
            chunk_size=1000,
        )
        
        print(f"Dataset size: {dataloader.dataset.total_rows} rows")
        print(f"Approximate number of batches: {dataloader.dataset.total_rows // args.batch_size}\n")
        
        # Load and display batches
        for batch_idx, (X, policy, R, WDL) in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
            
            print(f"Batch {batch_idx + 1}:")
            print("=" * 60)
            print(f"  X shape: {X.shape} (dtype: {X.dtype})")
            print(f"  policy shape: {policy.shape} (dtype: {policy.dtype})")
            print(f"  R shape: {R.shape} (dtype: {R.dtype})")
            print(f"  WDL shape: {WDL.shape} (dtype: {WDL.dtype})")
            print(f"  Sample values:")
            print(f"    policy[0]: {policy[0].item()}")
            print(f"    R[0]: {R[0].item()}")
            print(f"    WDL[0]: {WDL[0].item()}")
            print()
        
        print("Dataloader test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

