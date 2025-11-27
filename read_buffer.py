#!/usr/bin/env python3
"""Read and display the binary self-play buffer."""

import struct
import sys
from pathlib import Path

def read_binary_buffer(path: Path, max_rows: int = 10):
    """Read and display binary buffer contents."""
    if not path.exists():
        print(f"File not found: {path}")
        return
    
    with path.open("rb") as f:
        content = f.read()
    
    offset = 0
    size = len(content)
    
    fens = []
    policies = []
    values = []
    
    print(f"Reading {path} ({size:,} bytes)")
    print("=" * 80)
    
    print(f"Reading {path} ({size:,} bytes)")
    print("=" * 80)
    
    RECORD_SIZE = 128
    
    row_count = 0
    while offset + RECORD_SIZE <= size:
        # Read FEN (100 bytes)
        fen_bytes = content[offset:offset+100]
        try:
            fen = fen_bytes.split(b'\0', 1)[0].decode("utf-8")
        except:
            fen = "<invalid>"
        offset += 100
        
        # Read Policy (16 bytes)
        pol_bytes = content[offset:offset+16]
        try:
            policy = pol_bytes.split(b'\0', 1)[0].decode("utf-8")
        except:
            policy = "<invalid>"
        offset += 16
        
        # Read Value (12 bytes)
        win, draw, loss = struct.unpack_from("fff", content, offset)
        offset += 12
        
        fens.append(fen)
        policies.append(policy)
        values.append((win, draw, loss))
        
        if row_count < max_rows:
            print(f"\nRow {row_count + 1}:")
            print(f"  FEN:    {fen}")
            print(f"  Policy: {policy}")
            print(f"  WDL:    W={win:.3f}, D={draw:.3f}, L={loss:.3f}")
        
        row_count += 1
    
    print("\n" + "=" * 80)
    print(f"Total rows: {len(fens):,}")
    
    # Statistics
    print("\nStatistics:")
    print("-" * 40)
    
    # Duplicates
    unique_fens = set(fens)
    print(f"Unique FENs: {len(unique_fens):,} ({len(fens) - len(unique_fens):,} duplicates)")
    
    # Knight promotions
    knight_promotions = [p for p in policies if "n" in p and len(p) == 5] # e.g. a7a8n
    print(f"Knight promotions: {len(knight_promotions)}")
    
    # WDL Distribution
    wins = [v for v in values if v[0] > v[1] and v[0] > v[2]]
    draws = [v for v in values if v[1] > v[0] and v[1] > v[2]]
    losses = [v for v in values if v[2] > v[0] and v[2] > v[1]]
    
    print(f"Outcomes (Argmax):")
    print(f"  Wins:   {len(wins):,} ({len(wins)/len(fens)*100:.1f}%)")
    print(f"  Draws:  {len(draws):,} ({len(draws)/len(fens)*100:.1f}%)")
    print(f"  Losses: {len(losses):,} ({len(losses)/len(fens)*100:.1f}%)")
    
    # High confidence
    high_wins = [v for v in values if v[0] > 0.7]
    high_losses = [v for v in values if v[2] > 0.7]
    print(f"High confidence (>0.7):")
    print(f"  Wins:   {len(high_wins):,}")
    print(f"  Losses: {len(high_losses):,}")
    
    # Color bias in wins/losses
    # Argmax0 (Wins) - check if mostly White or Black to move
    wins_indices = [i for i, v in enumerate(values) if v[0] > v[1] and v[0] > v[2]]
    if wins_indices:
        white_wins = len([i for i in wins_indices if " w " in fens[i]])
        print(f"Win bias: {white_wins/len(wins_indices)*100:.1f}% White to move")
        
    losses_indices = [i for i, v in enumerate(values) if v[2] > v[0] and v[2] > v[1]]
    if losses_indices:
        black_losses = len([i for i in losses_indices if " b " in fens[i]]) # If Loss and Black to move, means White won
        # Actually, if Loss (v[2] high), it means side to move lost.
        # So if Black to move and Loss, Black lost (White won).
        print(f"Loss bias: {black_losses/len(losses_indices)*100:.1f}% Black to move")

if __name__ == "__main__":
    buffer_path = Path("src/self_play/selfplay_buffer.bin")
    if len(sys.argv) > 1:
        buffer_path = Path(sys.argv[1])
    
    max_rows = 100
    if len(sys.argv) > 2:
        max_rows = int(sys.argv[2])
    
    read_binary_buffer(buffer_path, max_rows)
