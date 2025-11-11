#!/usr/bin/env python3
"""
Memory-efficient parquet shuffling using chunked batch approach.

Reads file in chunks, shuffles within each chunk, yields batches of 1024 rows.
Shows indices for each batch and measures throughput (rows/second).
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def yield_shuffled_batches(parquet_path, chunk_size=10000, batch_size=1024, seed=42):
    """
    Memory-bounded chunked shuffling that yields batches.
    
    Reads file in chunks, shuffles within each chunk, yields batches of specified size.
    This provides local shuffling which is memory-efficient and sufficient for ML training.
    
    Memory usage: O(chunk_size) - truly bounded!
    
    Args:
        parquet_path: Path to parquet file
        chunk_size: Number of rows per chunk to read and shuffle
        batch_size: Number of rows per yielded batch
        seed: Random seed
        
    Yields:
        (batch_indices, batch_dataframe) tuples
        - batch_indices: numpy array of original file indices for rows in this batch
        - batch_dataframe: pandas DataFrame with batch_size rows (or fewer for last batch)
    """
    parquet_file = pq.ParquetFile(parquet_path)
    rng = np.random.RandomState(seed)
    
    current_chunk_rows = []
    current_chunk_indices = []
    file_position = 0
    
    # Read file in chunks
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        batch_df = batch.to_pandas()
        
        # Create shuffled indices for this chunk
        chunk_indices = np.arange(len(batch_df))
        rng.shuffle(chunk_indices)
        
        # Add shuffled rows to buffer
        for local_idx in chunk_indices:
            current_chunk_rows.append(batch_df.iloc[local_idx])
            current_chunk_indices.append(file_position + local_idx)
            
            # Yield batch when we have enough rows
            if len(current_chunk_rows) >= batch_size:
                yield_df = pd.DataFrame(current_chunk_rows[:batch_size])
                yield_indices = np.array(current_chunk_indices[:batch_size])
                
                yield yield_indices, yield_df
                
                # Remove yielded rows from buffer
                current_chunk_rows = current_chunk_rows[batch_size:]
                current_chunk_indices = current_chunk_indices[batch_size:]
        
        file_position += len(batch_df)
        del batch_df
        del batch
    
    # Yield remaining rows as final batch
    if len(current_chunk_rows) > 0:
        yield_df = pd.DataFrame(current_chunk_rows)
        yield_indices = np.array(current_chunk_indices)
        yield yield_indices, yield_df


def main():
    parser = argparse.ArgumentParser(
        description='Shuffle parquet file in batches and measure throughput'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/training_data.parquet',
        help='Input parquet file path (default: data/training_data.parquet)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Maximum number of batches to process (default: all batches)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Chunk size for reading and shuffling (default: 10000). '
             'Larger = better shuffle quality but more memory.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Number of rows per yielded batch (default: 1024)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: write shuffled data to output parquet file'
    )
    parser.add_argument(
        '--show-indices',
        action='store_true',
        help='Show the first few indices of each batch (verbose)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    try:
        parquet_file = pq.ParquetFile(args.input)
        total_rows = parquet_file.metadata.num_rows
    except Exception as e:
        print(f"Error: Failed to open parquet file {args.input}: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 80)
    print(f"Parquet Batch Shuffling Performance Test")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Total rows: {total_rows:,}")
    print(f"Chunk size: {args.chunk_size:,} rows")
    print(f"Batch size: {args.batch_size} rows")
    print(f"Random seed: {args.seed}")
    print()
    print("Note: Chunked shuffling provides local shuffling within chunks")
    print(f"      Memory usage: ~{args.chunk_size} rows at a time")
    print()
    
    # Create batch generator
    batch_generator = yield_shuffled_batches(
        args.input, 
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Process batches and measure throughput
    print("Processing shuffled batches...")
    print("=" * 80)
    
    start_time = time.time()
    last_report_time = start_time
    batches_processed = 0
    rows_processed = 0
    rows_since_last_report = 0
    
    # Optional: write to output file
    writer = None
    
    try:
        for batch_indices, batch_df in batch_generator:
            batches_processed += 1
            batch_size = len(batch_df)
            rows_processed += batch_size
            rows_since_last_report += batch_size
            
            # Show indices if requested
            if args.show_indices:
                if batch_size <= 10:
                    print(f"Batch {batches_processed}: indices = {batch_indices.tolist()}")
                else:
                    print(f"Batch {batches_processed}: indices = [{batch_indices[0]}, {batch_indices[1]}, "
                          f"{batch_indices[2]}, ..., {batch_indices[-2]}, {batch_indices[-1]}] "
                          f"(total: {batch_size} rows)")
            
            # Stop if we've processed enough batches
            if args.max_batches and batches_processed >= args.max_batches:
                break
            
            # Write to output if specified
            if args.output:
                if writer is None:
                    # Initialize writer on first batch
                    batch_table = pa.Table.from_pandas(batch_df)
                    writer = pq.ParquetWriter(args.output, batch_table.schema)
                
                # Convert batch to table and write
                batch_table = pa.Table.from_pandas(batch_df)
                writer.write_table(batch_table)
            
            # Report progress every second
            current_time = time.time()
            if current_time - last_report_time >= 1.0:
                elapsed = current_time - start_time
                throughput = rows_processed / elapsed
                progress_pct = (rows_processed / total_rows) * 100
                
                print(f"Batches: {batches_processed:,} | "
                      f"Rows: {rows_processed:,}/{total_rows:,} ({progress_pct:.1f}%) | "
                      f"Throughput: {throughput:.1f} rows/sec | "
                      f"Last second: {rows_since_last_report} rows")
                
                last_report_time = current_time
                rows_since_last_report = 0
    
    finally:
        if writer is not None:
            writer.close()
    
    # Final statistics
    total_time = time.time() - start_time
    avg_throughput = rows_processed / total_time if total_time > 0 else 0
    
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total batches processed: {batches_processed:,}")
    print(f"Total rows processed: {rows_processed:,}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average throughput: {avg_throughput:.1f} rows/second")
    print(f"Time per batch: {(total_time / batches_processed) * 1000:.3f} ms")
    print(f"Time per row: {(total_time / rows_processed) * 1000:.3f} ms")
    
    if args.output:
        print(f"\nShuffled data written to: {args.output}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
