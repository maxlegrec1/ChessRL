import os
import random
from typing import Iterator, Optional, Dict, Any

import bulletchess
import pandas as pd
import pyarrow.parquet as pq


def _load_hf_openings(split: str = "train"):

    from datasets import load_dataset  # type: ignore
    _DATASET = load_dataset("Lichess/chess-openings", split=split)
    return _DATASET


def _epd_to_fen(epd: str) -> Optional[str]:
    # EPD: fields similar to FEN but usually 4 tokens + operations; convert to FEN by taking first 4 and appending clocks
    parts = epd.strip().split()
    if len(parts) < 4:
        return None
    fen4 = " ".join(parts[:4])
    fen = f"{fen4} 0 1"
    try:
        # Validate
        _ = bulletchess.Board.from_fen(fen)
        return fen
    except Exception:
        return None


def _load_parquet_openings(parquet_path: str):
    """Load parquet file metadata for openings."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    parquet_file = pq.ParquetFile(parquet_path)
    return parquet_file


def parquet_opening_fen_generator(parquet_path: Optional[str] = None, seed: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield random openings sampled from openings.parquet.

    Each item is a dict with:
      - 'fen': FEN string
      - 'uci': List[str] of UCI moves that lead to the FEN (may be empty)
    
    Args:
        parquet_path: Path to the parquet file. Defaults to src/generate/openings.parquet
        seed: Random seed for reproducibility
    """
    if parquet_path is None:
        # Default to src/generate/openings.parquet relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parquet_path = os.path.join(current_dir, "openings.parquet")
    
    parquet_file = _load_parquet_openings(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    
    if total_rows == 0:
        raise ValueError(f"Parquet file {parquet_path} is empty")
    
    rnd = random.Random(seed)
    
    while True:
        idx = rnd.randrange(0, total_rows)
        
        # Read a batch containing the row we want
        # Find which row group contains this index
        row_group_idx = 0
        rows_read = 0
        for i in range(parquet_file.metadata.num_row_groups):
            rg_meta = parquet_file.metadata.row_group(i)
            if idx < rows_read + rg_meta.num_rows:
                row_group_idx = i
                break
            rows_read += rg_meta.num_rows
        
        # Read the row group and extract the specific row
        table = parquet_file.read_row_group(row_group_idx)
        local_idx = idx - rows_read
        
        if local_idx < 0 or local_idx >= table.num_rows:
            # Shouldn't happen, but skip if it does
            continue
        
        # Convert to pandas for easier row access
        df = table.to_pandas()
        row = df.iloc[local_idx]
        
        fen = str(row["fen"])
        if not fen:
            continue
        
        # Validate FEN
        try:
            _ = bulletchess.Board.from_fen(fen)
        except Exception:
            continue
        
        # Get moves list (column name is "moves" in parquet, map to "uci" for consistency)
        moves = row["moves"]
        # Convert numpy array to Python list if needed
        if hasattr(moves, 'tolist'):
            moves = moves.tolist()
        elif not isinstance(moves, list):
            moves = []
        
        yield {"fen": fen, "uci": moves}


def hf_opening_fen_generator(split: str = "train", seed: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield random openings sampled from Lichess/chess-openings.

    Each item is a dict with:
      - 'fen': FEN string
      - 'uci': List[str] of UCI moves that lead to the FEN (may be empty)
    """
    ds = _load_hf_openings(split)
    rnd = random.Random(seed)
    n = len(ds)
    while True:
        idx = rnd.randrange(0, n)
        rec = ds[int(idx)]
        if not isinstance(rec, dict):
            continue
        epd = rec.get("epd")
        if not epd:
            continue
        fen = _epd_to_fen(epd)
        if fen is None:
            continue
        uci_str = rec.get("uci") or ""
        uci_moves = [mv for mv in uci_str.strip().split() if mv]
        yield {"fen": fen, "uci": uci_moves}


