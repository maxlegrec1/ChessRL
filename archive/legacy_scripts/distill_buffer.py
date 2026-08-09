#!/usr/bin/env python3
"""Distill a smaller binary buffer with balanced high/normal confidence samples."""

import struct
import sys
import random
from pathlib import Path

def distill_buffer(input_path: Path, output_path: Path, confidence_threshold: float = 0.7):
    """
    Distill buffer to keep:
    - All high-confidence samples (WDL[0] > threshold or WDL[2] > threshold)
    - Random sample of normal-confidence to achieve ~50/50 balance
    """
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return
    
    with input_path.open("rb") as f:
        content = f.read()
    
    size = len(content)
    RECORD_SIZE = 128
    
    high_confidence_records = []
    normal_confidence_records = []
    
    offset = 0
    while offset + RECORD_SIZE <= size:
        record = content[offset:offset + RECORD_SIZE]
        
        # Read Value (at offset 116, 12 bytes = 3 floats)
        win, draw, loss = struct.unpack_from("fff", record, 116)
        
        if win > confidence_threshold or loss > confidence_threshold:
            high_confidence_records.append(record)
        else:
            normal_confidence_records.append(record)
        
        offset += RECORD_SIZE
    
    print(f"Read {len(high_confidence_records) + len(normal_confidence_records):,} total records")
    print(f"  High confidence (>{confidence_threshold}): {len(high_confidence_records):,}")
    print(f"  Normal confidence: {len(normal_confidence_records):,}")
    
    # Sample normal confidence to match high confidence count
    target_normal = len(high_confidence_records)
    if len(normal_confidence_records) > target_normal:
        sampled_normal = random.sample(normal_confidence_records, target_normal)
    else:
        sampled_normal = normal_confidence_records
    
    print(f"\nSampled {len(sampled_normal):,} normal confidence records")
    
    # Combine and shuffle
    all_records = high_confidence_records + sampled_normal
    random.shuffle(all_records)
    
    print(f"Total output records: {len(all_records):,}")
    print(f"  High confidence: {len(high_confidence_records):,} ({len(high_confidence_records)/len(all_records)*100:.1f}%)")
    print(f"  Normal confidence: {len(sampled_normal):,} ({len(sampled_normal)/len(all_records)*100:.1f}%)")
    
    # Write output
    with output_path.open("wb") as f:
        for record in all_records:
            f.write(record)
    
    print(f"\nWritten to: {output_path}")
    print(f"Output size: {len(all_records) * RECORD_SIZE:,} bytes")


if __name__ == "__main__":
    input_path = Path("buffer_value.bin")
    output_path = Path("buffer_value_distilled.bin")
    threshold = 0.7
    
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    if len(sys.argv) > 3:
        threshold = float(sys.argv[3])
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Confidence threshold: {threshold}")
    print("=" * 60)
    
    distill_buffer(input_path, output_path, threshold)

