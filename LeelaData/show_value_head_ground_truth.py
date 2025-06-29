#!/usr/bin/env python3
"""
Script to extract all training data from LCZero chunks and save as parquet files.

This script reads all positions from training chunks and saves tuples of:
- FEN: Chess position in FEN notation
- WDL: Win-Draw-Loss probabilities (3-element vector)  
- Policy: Move probabilities (1858-element vector)

Data is saved in parquet files with 6M rows each for efficient storage and loading.

Usage:
    python show_value_head_ground_truth.py [path_to_training_files] [output_directory]
"""

import sys
import struct
import numpy as np
import gzip
import glob
import os
import pandas as pd
from tqdm import tqdm
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow is required for this script.")
    print("Please install it with: pip install pyarrow")
    sys.exit(1)

# Add the lczero-training tf directory to path to import chunkparser
sys.path.append('lczero-training/tf')
import chunkparser

def qd_to_wdl(q, d):
    """Convert Q (value) and D (draw probability) to WDL probabilities.
    
    This is the core conversion used in LCZero training.
    
    Args:
        q: Value from -1 (loss) to +1 (win)
        d: Draw probability from 0 to 1
        
    Returns:
        (win_prob, draw_prob, loss_prob) tuple
    """
    # Clamp values to valid ranges
    q = max(-1.0, min(1.0, q))
    d = max(0.0, min(1.0, d))
    
    # Convert to WDL
    w = 0.5 * (1.0 - d + q)  # Win probability
    l = 0.5 * (1.0 - d - q)  # Loss probability
    
    return (w, d, l)

def planes_to_board(planes_bytes):
    """Convert binary plane representation to a chess board.
    
    Args:
        planes_bytes: Raw bytes representing the 12 piece bit planes (96 bytes total)
        
    Returns:
        8x8 board array where each cell contains the piece character or '.'
    """
    # Chess piece mapping
    PIECES = "PNBRQKpnbrqk"  # Our pieces: PNBRQK, Their pieces: pnbrqk
    
    # Unpack the bit planes - 12 planes for pieces
    planes = np.unpackbits(np.frombuffer(planes_bytes, dtype=np.uint8)).astype(np.uint8)
    planes = np.reshape(planes, [12, 64])  # 12 planes of 64 squares each
    
    # Initialize empty board
    board = [['.' for _ in range(8)] for _ in range(8)]
    
    # Place pieces from all 12 planes
    for piece_idx in range(12):
        piece_char = PIECES[piece_idx]
        plane = planes[piece_idx]
        
        for square in range(64):
            if plane[square]:
                row = square // 8
                col = square % 8
                board[7-row][col] = piece_char  # Flip row order for display
    
    return board

def board_to_fen_position(board, us_black, castling_info, rule50_count):
    """Convert board representation to FEN string.
    
    Args:
        board: 8x8 board array
        us_black: Whether we are playing as black (1) or white (0)
        castling_info: Tuple of (us_ooo, us_oo, them_ooo, them_oo)
        rule50_count: Half-move clock for 50-move rule
        
    Returns:
        FEN string representing the position
    """
    # Build piece placement part of FEN
    fen_parts = []
    
    for row in board:
        fen_row = ""
        empty_count = 0
        
        for cell in row:
            if cell == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_parts.append(fen_row)
    
    piece_placement = '/'.join(fen_parts)
    
    # Active color
    active_color = 'b' if us_black else 'w'
    
    # Castling availability
    us_ooo, us_oo, them_ooo, them_oo = castling_info
    castling = ""
    if us_black:
        # We are black, so our castling rights are lowercase
        if us_oo: castling += "k"
        if us_ooo: castling += "q"
        if them_oo: castling += "K"
        if them_ooo: castling += "Q"
    else:
        # We are white, so our castling rights are uppercase
        if us_oo: castling += "K"
        if us_ooo: castling += "Q"
        if them_oo: castling += "k"
        if them_ooo: castling += "q"
    
    if not castling:
        castling = "-"
    
    # En passant target square (not available in this data format)
    en_passant = "-"
    
    # Half-move clock (rule50_count)
    halfmove_clock = str(rule50_count)
    
    # Full-move number (we don't have this info, so use 1)
    fullmove_number = "1"
    
    return f"{piece_placement} {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"

def extract_all_positions(filename):
    """Extract all positions from a training file.
    
    Args:
        filename: Path to .gz training file
        
    Yields:
        Dictionary with fen, wdl, and policy for each position
    """
    with gzip.open(filename, "rb") as chunk_file:
        chunkdata = chunk_file.read()
        if len(chunkdata) == 0:
            return
        
        version = chunkdata[0:4]
        record_size = chunkparser.struct_sizes.get(version, None)
        
        if record_size is None:
            print(f"Unknown version {version} in file {filename}")
            return
        
        n_chunks = len(chunkdata) // record_size
        
        for i in range(n_chunks):
            record_start = i * record_size
            record = chunkdata[record_start:record_start + record_size]
            
            # Parse the full record
            if version == chunkparser.V6_VERSION or version == chunkparser.V7_VERSION:
                # Pad to v7b size if needed for parsing
                if len(record) < chunkparser.v7b_struct.size:
                    record += b'\x00' * (chunkparser.v7b_struct.size - len(record))
                
                # Extract position information for FEN
                raw_planes = record[7440:7440 + 832]  # 832 bytes = 104 planes * 8 bytes each
                us_ooo = record[8272]
                us_oo = record[8273]
                them_ooo = record[8274]
                them_oo = record[8275]
                us_black = record[8276] & 0x80  # bit 7
                rule50_count = record[8277]
                
                # Extract first 12 planes (96 bytes) for pieces
                piece_planes = raw_planes[:96]  # 12 planes * 8 bytes each = 96 bytes
                board = planes_to_board(piece_planes)
                castling_info = (us_ooo, us_oo, them_ooo, them_oo)
                fen = board_to_fen_position(board, us_black, castling_info, rule50_count)
                
                # Extract policy probabilities (1858 floats starting at byte 8)
                policy_bytes = record[8:8 + 1858*4]  # 1858 floats * 4 bytes each
                policy_probs = list(struct.unpack('1858f', policy_bytes))
                
                # Parse using chunkparser to get the final training targets
                parsed = chunkparser.convert_v7b_to_tuple(record)
                
                # Extract the WDL values that are actually used for training (root WDL)
                root_wdl_bytes = parsed[3]  # parsed[3] is root WDL used for value head
                root_wdl = list(struct.unpack("fff", root_wdl_bytes))
                
                yield {
                    'fen': fen,
                    'wdl': root_wdl,
                    'policy': policy_probs
                }

class ParquetWriter:
    """Streaming parquet writer that handles batching and file rotation."""
    
    def __init__(self, output_dir, positions_per_file=6_000_000, write_batch_size=10_000):
        self.output_dir = output_dir
        self.positions_per_file = positions_per_file
        self.write_batch_size = write_batch_size
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.current_file_num = 0
        self.current_file_positions = 0
        self.current_writer = None
        self.current_batch = []
        self.total_positions = 0
        
        # Define schema
        self.schema = pa.schema([
            pa.field('fen', pa.string()),
            pa.field('wdl', pa.list_(pa.float32(), 3)),
            pa.field('policy', pa.list_(pa.float32(), 1858))
        ])
        
    def _get_filename(self, file_num):
        return os.path.join(self.output_dir, f"training_data_batch_{file_num:06d}.parquet")
    
    def _start_new_file(self):
        """Start writing to a new parquet file."""
        if self.current_writer:
            self.current_writer.close()
        
        filename = self._get_filename(self.current_file_num)
        self.current_writer = pq.ParquetWriter(filename, self.schema)
        self.current_file_positions = 0
        print(f"Started new parquet file: {filename}")
    
    def _write_batch(self):
        """Write current batch to file."""
        if not self.current_batch:
            return
            
        if not self.current_writer:
            self._start_new_file()
        
        # Convert batch to arrow table
        df = pd.DataFrame(self.current_batch)
        table = pa.Table.from_pandas(df, schema=self.schema)
        
        # Write to parquet
        self.current_writer.write_table(table)
        
        batch_size = len(self.current_batch)
        self.current_file_positions += batch_size
        self.total_positions += batch_size
        
        # Clear batch
        self.current_batch = []
        
        # Check if we need to rotate to a new file
        if self.current_file_positions >= self.positions_per_file:
            print(f"Completed file {self.current_file_num} with {self.current_file_positions:,} positions")
            self.current_writer.close()
            self.current_writer = None
            self.current_file_num += 1
    
    def add_position(self, position_data):
        """Add a single position to the writer."""
        self.current_batch.append(position_data)
        
        # Write batch when it reaches the target size
        if len(self.current_batch) >= self.write_batch_size:
            self._write_batch()
    
    def finalize(self):
        """Write any remaining data and close files."""
        if self.current_batch:
            self._write_batch()
        
        if self.current_writer:
            print(f"Completed final file {self.current_file_num} with {self.current_file_positions:,} positions")
            self.current_writer.close()
        
        return self.total_positions, self.current_file_num + 1

def process_all_files(training_files, output_dir):
    """Process all training files and save data in parquet format."""
    
    print(f"Processing {len(training_files)} training files...")
    print(f"Will save parquet files every 6M positions to: {output_dir}")
    print("Writing data continuously in 10K position batches to manage memory usage.")
    
    writer = ParquetWriter(output_dir)
    
    for file_idx, filename in enumerate(training_files):
        print(f"\n[{file_idx+1}/{len(training_files)}] Processing: {os.path.basename(filename)}")
        
        try:
            file_positions = 0
            for position_data in tqdm(extract_all_positions(filename), desc="Extracting positions"):
                writer.add_position(position_data)
                file_positions += 1
            
            print(f"  Extracted {file_positions:,} positions from this file")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    # Finalize writing
    total_positions, total_files = writer.finalize()
    
    print(f"\n=== Summary ===")
    print(f"Total positions processed: {total_positions:,}")
    print(f"Total parquet files created: {total_files}")
    print(f"Data saved to: {output_dir}")
    
    return total_positions

def main():
    """Main function to extract all training data and save as parquet files."""
    
    print("=== LCZero Training Data Extraction to Parquet ===")
    print("\nThis script extracts all positions from LCZero training chunks and saves them as parquet files.")
    print("Each position contains: FEN, WDL (3-element vector), Policy (1858-element vector)")
    print("Data is saved in batches of 6M positions for efficient storage and loading.")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\nUsage: python show_value_head_ground_truth.py <path_to_training_files> [output_directory]")
        print("Example: python show_value_head_ground_truth.py T80_extracted/training-run1-test80-20220404-0417/ output_parquet/")
        return
    
    search_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_parquet"
    
    # Find all .gz training files
    if os.path.isdir(search_path):
        training_files = glob.glob(os.path.join(search_path, "*.gz"))
    else:
        # Single file
        training_files = [search_path] if search_path.endswith('.gz') else []
    
    if not training_files:
        print(f"\nNo .gz training files found in {search_path}")
        print("Please provide a directory containing .gz training files or a single .gz file.")
        return
    
    print(f"\nFound {len(training_files)} training files to process")
    print(f"Output directory: {output_dir}")
    
    # Process all files
    try:
        total_positions = process_all_files(training_files, output_dir)
        
        print(f"\n=== Extraction Complete ===")
        print(f"Successfully extracted {total_positions:,} positions")
        print(f"Parquet files saved to: {output_dir}")
        print("\nData format:")
        print("- fen: Chess position in FEN notation (string)")
        print("- wdl: Win-Draw-Loss probabilities [W, D, L] (3 floats)")
        print("- policy: Move probabilities (1858 floats)")
        
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user")
    except Exception as e:
        print(f"\nError during extraction: {e}")
        raise

if __name__ == "__main__":
    main() 