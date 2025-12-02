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
    
    size = len(content)
    
    fens = []
    policies = []
    values = []
    
    print(f"Reading {path} ({size:,} bytes)")
    print("=" * 80)
    
    print(f"Reading {path} ({size:,} bytes)")
    print("=" * 80)
    
    RECORD_SIZE = 128
    offset = 0
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
    if len(fens) > 0:
        wins = [v for v in values if v[0] > v[1] and v[0] > v[2]]
        draws = [v for v in values if v[1] > v[0] and v[1] > v[2]]
        losses = [v for v in values if v[2] > v[0] and v[2] > v[1]]
        
        print(f"Outcomes (Argmax):")
        print(f"  Wins:   {len(wins):,} ({len(wins)/len(fens)*100:.1f}%)")
        print(f"  Draws:  {len(draws):,} ({len(draws)/len(fens)*100:.1f}%)")
        print(f"  Losses: {len(losses):,} ({len(losses)/len(fens)*100:.1f}%)")
    else:
        print("Outcomes (Argmax): No data to analyze")
    
    # High confidence
    if len(fens) > 0:
        high_win_indices = [i for i, v in enumerate(values) if v[0] > 0.7]
        high_loss_indices = [i for i, v in enumerate(values) if v[2] > 0.7]
        print(f"High confidence (>0.7):")
        print(f"  Wins:   {len(high_win_indices):,}")
        print(f"  Losses: {len(high_loss_indices):,}")
        
        # Print sample high-confidence wins
        print(f"\nSample high-confidence wins (win_prob > 0.7):")
        print("-" * 60)
        for i, idx in enumerate(high_win_indices[:10]):
            fen = fens[idx]
            w, d, l = values[idx]
            stm = "White" if " w " in fen else "Black"
            print(f"  {i+1}. [{stm} to move] WDL=[{w:.3f}, {d:.3f}, {l:.3f}]")
            print(f"     FEN: {fen}")
        
        # Print sample high-confidence losses
        print(f"\nSample high-confidence losses (loss_prob > 0.7):")
        print("-" * 60)
        for i, idx in enumerate(high_loss_indices[:10]):
            fen = fens[idx]
            w, d, l = values[idx]
            stm = "White" if " w " in fen else "Black"
            print(f"  {i+1}. [{stm} to move] WDL=[{w:.3f}, {d:.3f}, {l:.3f}]")
            print(f"     FEN: {fen}")
    
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
    
    # Bias breakdown by confidence level
    print(f"\n" + "=" * 60)
    print("BIAS BREAKDOWN BY CONFIDENCE LEVEL:")
    print("=" * 60)
    
    # For wins (argmax = 0)
    if wins_indices:
        high_conf_wins = [i for i in wins_indices if values[i][0] > 0.5]
        low_conf_wins = [i for i in wins_indices if values[i][0] <= 0.5]
        
        if high_conf_wins:
            white_high = len([i for i in high_conf_wins if " w " in fens[i]])
            print(f"\nHigh-confidence wins (WDL[0] > 0.5): {len(high_conf_wins):,}")
            print(f"  White to move: {white_high:,} ({white_high/len(high_conf_wins)*100:.1f}%)")
            print(f"  Black to move: {len(high_conf_wins)-white_high:,} ({(len(high_conf_wins)-white_high)/len(high_conf_wins)*100:.1f}%)")
        
        if low_conf_wins:
            white_low = len([i for i in low_conf_wins if " w " in fens[i]])
            print(f"\nLow-confidence wins (WDL[0] <= 0.5): {len(low_conf_wins):,}")
            print(f"  White to move: {white_low:,} ({white_low/len(low_conf_wins)*100:.1f}%)")
            print(f"  Black to move: {len(low_conf_wins)-white_low:,} ({(len(low_conf_wins)-white_low)/len(low_conf_wins)*100:.1f}%)")
            
            # Show sample low-confidence wins
            print(f"\n  Sample low-confidence 'wins' (first 5):")
            for i, idx in enumerate(low_conf_wins[:5]):
                fen = fens[idx]
                w, d, l = values[idx]
                stm = "White" if " w " in fen else "Black"
                print(f"    {i+1}. [{stm}] WDL=[{w:.4f}, {d:.4f}, {l:.4f}]")
    
    # For losses (argmax = 2)
    if losses_indices:
        high_conf_losses = [i for i in losses_indices if values[i][2] > 0.5]
        low_conf_losses = [i for i in losses_indices if values[i][2] <= 0.5]
        
        if high_conf_losses:
            black_high = len([i for i in high_conf_losses if " b " in fens[i]])
            print(f"\nHigh-confidence losses (WDL[2] > 0.5): {len(high_conf_losses):,}")
            print(f"  Black to move: {black_high:,} ({black_high/len(high_conf_losses)*100:.1f}%)")
            print(f"  White to move: {len(high_conf_losses)-black_high:,} ({(len(high_conf_losses)-black_high)/len(high_conf_losses)*100:.1f}%)")
        
        if low_conf_losses:
            black_low = len([i for i in low_conf_losses if " b " in fens[i]])
            print(f"\nLow-confidence losses (WDL[2] <= 0.5): {len(low_conf_losses):,}")
            print(f"  Black to move: {black_low:,} ({black_low/len(low_conf_losses)*100:.1f}%)")
            print(f"  White to move: {len(low_conf_losses)-black_low:,} ({(len(low_conf_losses)-black_low)/len(low_conf_losses)*100:.1f}%)")
            
            # Show sample low-confidence losses
            print(f"\n  Sample low-confidence 'losses' (first 5):")
            for i, idx in enumerate(low_conf_losses[:5]):
                fen = fens[idx]
                w, d, l = values[idx]
                stm = "White" if " w " in fen else "Black"
                print(f"    {i+1}. [{stm}] WDL=[{w:.4f}, {d:.4f}, {l:.4f}]")

if __name__ == "__main__":
    buffer_path = Path("buffer_value.bin")
    if len(sys.argv) > 1:
        buffer_path = Path(sys.argv[1])
    
    max_rows = 100
    if len(sys.argv) > 2:
        max_rows = int(sys.argv[2])
    
    read_binary_buffer(buffer_path, max_rows)
