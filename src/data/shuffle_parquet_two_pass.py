#!/usr/bin/env python3
"""
Two-pass bucket shuffle for parquet files - FAST and memory-efficient.

Strategy:
  Pass 1: Read data, randomly assign to buckets, write buckets to disk
  Pass 2: Read each bucket, shuffle within bucket, yield batches

Memory usage: Only 1-2 chunks at a time.
Shuffle quality: Near-perfect in just 2 passes!
"""

import argparse
import os
import sys
import tempfile
import time
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def two_pass_shuffle(input_path, output_path, num_buckets, chunk_size, 
                     batch_size=1024, seed=42, show_progress=True, skip_pass2=False):
    """
    Perform two-pass bucket shuffle on parquet file.
    
    Args:
        input_path: Input parquet file path
        output_path: Output parquet file path (can be None for dry run)
        num_buckets: Number of buckets to use
        chunk_size: Chunk size for reading input
        batch_size: Batch size for yielding/writing
        seed: Random seed
        show_progress: Whether to show progress messages
        skip_pass2: If True, skip within-bucket shuffle (faster, but preserves local order)
        
    Yields:
        (batch_indices, batch_dataframe) tuples
    """
    rng = np.random.RandomState(seed)
    
    # Create temporary directory for buckets
    temp_dir = tempfile.mkdtemp(prefix='shuffle_buckets_')
    
    try:
        # =====================================================================
        # PASS 1: Assign rows to random buckets and write to disk
        # =====================================================================
        if show_progress:
            print("=" * 80)
            print("PASS 1: Assigning rows to random buckets")
            print("=" * 80)
        
        parquet_file = pq.ParquetFile(input_path)
        total_rows = parquet_file.metadata.num_rows
        schema = parquet_file.schema_arrow
        
        if show_progress:
            print(f"Total rows: {total_rows:,}")
            print(f"Number of buckets: {num_buckets}")
            print(f"Temporary directory: {temp_dir}")
            print()
        
        # Create bucket writers
        bucket_writers = {}
        bucket_counts = [0] * num_buckets
        
        rows_processed = 0
        start_time = time.time()
        last_report = start_time
        
        # Read input in chunks
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            batch_df = batch.to_pandas()
            
            # Assign each row to a random bucket
            bucket_assignments = rng.randint(0, num_buckets, size=len(batch_df))
            
            # Group rows by bucket
            for bucket_id in range(num_buckets):
                mask = bucket_assignments == bucket_id
                if not mask.any():
                    continue
                
                bucket_rows = batch_df[mask]
                bucket_counts[bucket_id] += len(bucket_rows)
                
                # Write to bucket file
                if bucket_id not in bucket_writers:
                    bucket_path = os.path.join(temp_dir, f'bucket_{bucket_id:06d}.parquet')
                    bucket_writers[bucket_id] = pq.ParquetWriter(bucket_path, schema)
                
                bucket_table = pa.Table.from_pandas(bucket_rows, schema=schema)
                bucket_writers[bucket_id].write_table(bucket_table)
            
            rows_processed += len(batch_df)
            
            # Progress report
            if show_progress:
                current_time = time.time()
                if current_time - last_report >= 1.0:
                    elapsed = current_time - start_time
                    throughput = rows_processed / elapsed
                    pct = (rows_processed / total_rows) * 100
                    print(f"  Pass 1: {rows_processed:,}/{total_rows:,} rows "
                          f"({pct:.1f}%) | {throughput:.0f} rows/sec")
                    last_report = current_time
            
            del batch_df
            del batch
        
        # Close all bucket writers
        for writer in bucket_writers.values():
            writer.close()
        
        if show_progress:
            elapsed = time.time() - start_time
            print(f"\n  Pass 1 complete: {rows_processed:,} rows in {elapsed:.1f}s")
            print(f"  Bucket sizes: min={min(bucket_counts):,}, "
                  f"max={max(bucket_counts):,}, "
                  f"avg={sum(bucket_counts)//len(bucket_counts):,}")
            print()
        
        # =====================================================================
        # PASS 2: Read buckets, shuffle within each, and yield/write
        # =====================================================================
        if show_progress:
            print("=" * 80)
            if skip_pass2:
                print("PASS 2: Reading buckets and writing output (skipping within-bucket shuffle)")
            else:
                print("PASS 2: Shuffling within buckets and writing output")
            print("=" * 80)
        
        output_writer = None
        if output_path:
            output_writer = pq.ParquetWriter(output_path, schema)
        
        batches_yielded = 0
        rows_yielded = 0
        current_batch_rows = []
        current_batch_indices = []
        current_output_position = 0
        
        start_time = time.time()
        last_report = start_time
        
        for bucket_id in range(num_buckets):
            bucket_path = os.path.join(temp_dir, f'bucket_{bucket_id:06d}.parquet')
            
            if not os.path.exists(bucket_path):
                continue
            
            # Read bucket
            bucket_file = pq.ParquetFile(bucket_path)
            bucket_df = bucket_file.read().to_pandas()
            
            # Shuffle rows within this bucket (unless skip_pass2 is True)
            if not skip_pass2:
                shuffle_indices = np.arange(len(bucket_df))
                rng.shuffle(shuffle_indices)
                bucket_df = bucket_df.iloc[shuffle_indices].reset_index(drop=True)
            
            # Add to batch buffer and yield when ready
            for idx in range(len(bucket_df)):
                current_batch_rows.append(bucket_df.iloc[idx])
                current_batch_indices.append(current_output_position)
                current_output_position += 1
                
                if len(current_batch_rows) >= batch_size:
                    # Create batch
                    batch_df = pd.DataFrame(current_batch_rows)
                    batch_indices = np.array(current_batch_indices)
                    
                    # Write to output if specified
                    if output_writer:
                        batch_table = pa.Table.from_pandas(batch_df, schema=schema)
                        output_writer.write_table(batch_table)
                    
                    # Yield batch
                    yield batch_indices, batch_df
                    
                    batches_yielded += 1
                    rows_yielded += len(batch_df)
                    
                    # Clear buffer
                    current_batch_rows = []
                    current_batch_indices = []
                    
                    # Progress report
                    if show_progress:
                        current_time = time.time()
                        if current_time - last_report >= 1.0:
                            elapsed = current_time - start_time
                            throughput = rows_yielded / elapsed
                            pct = (rows_yielded / total_rows) * 100
                            print(f"  Pass 2: {rows_yielded:,}/{total_rows:,} rows "
                                  f"({pct:.1f}%) | {throughput:.0f} rows/sec | "
                                  f"{batches_yielded:,} batches")
                            last_report = current_time
            
            del bucket_df
        
        # Yield final partial batch
        if len(current_batch_rows) > 0:
            batch_df = pd.DataFrame(current_batch_rows)
            batch_indices = np.array(current_batch_indices)
            
            if output_writer:
                batch_table = pa.Table.from_pandas(batch_df, schema=schema)
                output_writer.write_table(batch_table)
            
            yield batch_indices, batch_df
            batches_yielded += 1
            rows_yielded += len(batch_df)
        
        # Close output writer
        if output_writer:
            output_writer.close()
        
        if show_progress:
            elapsed = time.time() - start_time
            print(f"\n  Pass 2 complete: {rows_yielded:,} rows in {elapsed:.1f}s")
            print()
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            if show_progress:
                print(f"Cleaned up temporary directory: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Two-pass bucket shuffle for parquet files (FAST method)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='src/data/training_data.parquet',
        help='Input parquet file path (default: src/data/training_data.parquet)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='src/data/shuffled_data.parquet',
        help='Output parquet file path (default: None = dry run)'
    )
    parser.add_argument(
        '--buckets',
        type=int,
        default=None,
        help='Number of buckets (default: auto = total_rows / chunk_size / 10)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Chunk size for reading (default: 10,000)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for output (default: 1024)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Maximum number of batches to process (for testing)'
    )
    parser.add_argument(
        '--show-indices',
        action='store_true',
        help='Show sample indices for each batch (verbose)'
    )
    parser.add_argument(
        '--skip-pass2',
        action='store_true',
        help='Skip Pass 2 (within-bucket shuffle). Faster but preserves local order patterns.'
    )
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Auto-calculate number of buckets
    if args.buckets is None:
        parquet_file = pq.ParquetFile(args.input)
        total_rows = parquet_file.metadata.num_rows
        # Use ~100 rows per bucket on average
        args.buckets = max(100, total_rows // args.chunk_size // 10)
    
    print("=" * 80)
    print("TWO-PASS BUCKET SHUFFLE")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output or '(dry run - no output)'}")
    print(f"Buckets: {args.buckets}")
    print(f"Chunk size: {args.chunk_size:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print()
    
    # Perform shuffle
    start_time = time.time()
    batches_seen = 0
    
    try:
        for batch_indices, batch_df in two_pass_shuffle(
            args.input,
            args.output,
            num_buckets=args.buckets,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            seed=args.seed,
            show_progress=True,
            skip_pass2=args.skip_pass2
        ):
            batches_seen += 1
            
            if args.show_indices and batches_seen <= 5:
                if len(batch_indices) <= 10:
                    print(f"Batch {batches_seen}: indices = {batch_indices.tolist()}")
                else:
                    print(f"Batch {batches_seen}: indices = "
                          f"[{batch_indices[0]}, {batch_indices[1]}, ..., "
                          f"{batch_indices[-2]}, {batch_indices[-1]}]")
            
            if args.max_batches and batches_seen >= args.max_batches:
                print(f"\nStopping after {args.max_batches} batches (for testing)")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Total batches: {batches_seen:,}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per batch: {(total_time / batches_seen * 1000):.2f} ms")
    
    if args.output:
        print(f"\nShuffled data written to: {args.output}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()

