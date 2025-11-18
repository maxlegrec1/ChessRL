#!/usr/bin/env python3
"""
Shuffle the rows of adversarial_moves.parquet.

Reads the parquet file, shuffles all rows randomly, and writes it back.
"""

import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def shuffle_parquet(input_path: str, output_path: str = None, seed: int = 42):
    """
    Shuffle the rows of a parquet file.
    
    Args:
        input_path: Path to the input parquet file
        output_path: Path to the output parquet file. If None, overwrites input file.
        seed: Random seed for reproducibility
    """
    if output_path is None:
        output_path = input_path
    
    print(f"Reading parquet file: {input_path}")
    
    # Read the entire parquet file into a pandas DataFrame
    df = pq.read_table(input_path).to_pandas()
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Shuffle the DataFrame rows
    print("Shuffling rows...")
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Write back to parquet file
    print(f"Writing shuffled data to: {output_path}")
    df_shuffled.to_parquet(output_path, index=False)
    
    print(f"Done! Shuffled {len(df_shuffled)} rows.")


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle the rows of adversarial_moves.parquet"
    )
    parser.add_argument(
        "--input",
        default="adversarial_moves.parquet",
        help="Input parquet file path (default: adversarial_moves.parquet)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet file path (default: overwrites input file)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    shuffle_parquet(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()

