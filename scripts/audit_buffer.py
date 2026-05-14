"""Thorough audit of a self-play buffer.bin.

Reports target-quality diagnostics that bear on the policy/value-noise
discussion: how decisive MCTS-WDL is, how often it disagrees with the
game outcome, policy-move diversity, plies-remaining distribution.

The on-disk record layout is the one written by cpp/selfplay.cpp and read
by src/train/dataset.py:

    [  0,100)  fen            (NUL-terminated utf8)
    [100,116)  policy_move    (NUL-terminated utf8 UCI, argmax over MCTS visits)
    [116,128)  value_outcome  (3x f32, W/D/L from final game result, side-to-move)
    [128,140)  value_mcts     (3x f32, W/D/L from in-tree MCTS root)
    [140,144)  plies_remaining (f32, plies until game end from this pos)
    [144,148)  backed_plies    (f32, MCTS-backed plies-remaining target)

Usage:
    uv run python scripts/audit_buffer.py \\
        --buffer artifacts/runs/<id>/buffer/buffer.bin \\
        --sample 200000
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RECORD_SIZE = 832
FEN_OFFSET = 0
POLICY_OFFSET = 100
VALUE_OUTCOME_OFFSET = 116
VALUE_MCTS_OFFSET = 128
PLIES_REMAINING_OFFSET = 140
BACKED_PLIES_OFFSET = 144
PAST_REPETITION_OFFSET = 148
CURRENT_REPETITION_OFFSET = 155
PAST_BITBOARDS_OFFSET = 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sz = args.buffer.stat().st_size
    n = sz // RECORD_SIZE
    print(f"buffer: {args.buffer}")
    print(f"  size: {sz/1e6:.1f} MB  records: {n:,}  record_size: {RECORD_SIZE}")
    writes_path = args.buffer.with_suffix(".bin.writes")
    if writes_path.is_file():
        try:
            writes = int(writes_path.read_text().strip())
            print(f"  total writes (lifetime): {writes:,}  -> turned over {writes/n:.2f}x")
        except Exception:
            pass

    sample = min(args.sample, n)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n, sample, replace=False)
    idxs.sort()  # better page locality
    print(f"  sampling {sample:,} records...")

    content = args.buffer.read_bytes()

    vo = np.zeros((sample, 3), dtype=np.float32)
    vm = np.zeros((sample, 3), dtype=np.float32)
    plies = np.zeros(sample, dtype=np.float32)
    bplies = np.zeros(sample, dtype=np.float32)
    pol_strs: list[str] = []
    fen_stm = np.zeros(sample, dtype=np.int8)  # 1 if white-to-move
    # AlphaZero history fields.
    n_filled_slots = np.zeros(sample, dtype=np.int8)   # # of non-empty past slots
    past_rep_any = np.zeros(sample, dtype=np.int8)     # any past repetition flag set
    cur_rep = np.zeros(sample, dtype=np.int8)          # current repetition flag

    for i, idx in enumerate(idxs):
        o = int(idx) * RECORD_SIZE
        vo[i] = struct.unpack_from("fff", content, o + VALUE_OUTCOME_OFFSET)
        vm[i] = struct.unpack_from("fff", content, o + VALUE_MCTS_OFFSET)
        (plies[i],) = struct.unpack_from("f", content, o + PLIES_REMAINING_OFFSET)
        (bplies[i],) = struct.unpack_from("f", content, o + BACKED_PLIES_OFFSET)
        pol_bytes = content[o + POLICY_OFFSET : o + VALUE_OUTCOME_OFFSET]
        pol = pol_bytes.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        pol_strs.append(pol)
        fen_bytes = content[o + FEN_OFFSET : o + POLICY_OFFSET]
        fen = fen_bytes.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        parts = fen.split()
        fen_stm[i] = 1 if (len(parts) >= 2 and parts[1] == "w") else 0
        # History payload.
        rep_bytes = content[o + PAST_REPETITION_OFFSET :
                            o + PAST_REPETITION_OFFSET + 7]
        past_rep_any[i] = 1 if any(b for b in rep_bytes) else 0
        cur_rep[i] = int(content[o + CURRENT_REPETITION_OFFSET])
        # Count non-zero slots by scanning 96-byte chunks.
        n_filled = 0
        for slot in range(7):
            slot_start = o + PAST_BITBOARDS_OFFSET + slot * 96
            if any(content[slot_start : slot_start + 96]):
                n_filled += 1
        n_filled_slots[i] = n_filled

    print()
    print("=" * 70)
    print("VALUE_OUTCOME (final game result, side-to-move WDL)")
    print("=" * 70)
    nonzero = vo.sum(axis=1) > 0.0
    n_z = int((~nonzero).sum())
    cls = vo.argmax(axis=1)
    print(f"  zero/empty rows:           {n_z:,}  ({100*n_z/sample:.2f}%)")
    print(f"  W rows (argmax=0):         {100*((cls==0)&nonzero).mean():.2f}%")
    print(f"  D rows (argmax=1):         {100*((cls==1)&nonzero).mean():.2f}%")
    print(f"  L rows (argmax=2):         {100*((cls==2)&nonzero).mean():.2f}%")
    # check label is one-hot or soft
    onehot = ((vo == 1.0).any(axis=1) & (vo.sum(axis=1) == 1.0)).mean()
    print(f"  rows that look one-hot:    {100*onehot:.2f}%")

    print()
    print("=" * 70)
    print("VALUE_MCTS (in-tree root WDL)")
    print("=" * 70)
    sums = vm.sum(axis=1)
    print(f"  sum stats:  min={sums.min():.3f}  max={sums.max():.3f}  mean={sums.mean():.3f}")
    print(f"  rows summing to ~0:        {100*(sums<0.01).mean():.2f}%")
    valid_m = sums > 0.5
    vm_v = vm[valid_m] / sums[valid_m, None]
    max_p = vm_v.max(axis=1)
    arg_v = vm_v.argmax(axis=1)
    print(f"  on valid {valid_m.sum():,} rows:")
    print(f"    W rows: {100*(arg_v==0).mean():.2f}%   D: {100*(arg_v==1).mean():.2f}%   L: {100*(arg_v==2).mean():.2f}%")
    print(f"    max-class probability histogram:")
    for lo, hi in [(0.0,0.4), (0.4,0.5), (0.5,0.6), (0.6,0.7), (0.7,0.8), (0.8,0.9), (0.9,1.001)]:
        f = ((max_p>=lo)&(max_p<hi)).mean()
        print(f"      [{lo:.2f},{hi:.2f}):  {100*f:5.2f}%")
    # entropy
    eps = 1e-8
    ent = -(vm_v*np.log(vm_v+eps)).sum(axis=1) / np.log(3)
    print(f"    mean entropy (normed):   {ent.mean():.3f}  (0=one-hot, 1=uniform)")
    print(f"    frac near-uniform (max<0.40):  {100*(max_p<0.40).mean():.2f}%")
    print(f"    frac confident (max>=0.80):    {100*(max_p>=0.80).mean():.2f}%")

    print()
    print("=" * 70)
    print("AGREEMENT between value_outcome and value_mcts")
    print("=" * 70)
    both_valid = nonzero & (sums > 0.5)
    nbv = int(both_valid.sum())
    if nbv:
        agree = (cls[both_valid] == arg_v[(sums>0.5)[nonzero] if False else valid_m[nonzero] if False else valid_m & nonzero])
        # Re-compute aligned masks straightforwardly:
        outcome_cls = vo.argmax(axis=1)
        mcts_cls    = np.zeros(sample, dtype=np.int64)
        mcts_cls[valid_m] = vm_v.argmax(axis=1)
        m = nonzero & valid_m
        ag = (outcome_cls[m] == mcts_cls[m])
        print(f"  rows where both are present: {m.sum():,}  ({100*m.mean():.2f}%)")
        print(f"  argmax agree:        {100*ag.mean():.2f}%")
        # subdivide by outcome class
        for c, name in [(0,"W"),(1,"D"),(2,"L")]:
            sel = m & (outcome_cls == c)
            if sel.any():
                ag_c = (mcts_cls[sel] == c).mean()
                print(f"    outcome={name}: MCTS also {name} in {100*ag_c:.1f}% of {int(sel.sum()):,} rows")

    print()
    print("=" * 70)
    print("POLICY MOVE (single UCI argmax of MCTS visits)")
    print("=" * 70)
    nonempty = [p for p in pol_strs if p]
    print(f"  empty policy strings:   {sample - len(nonempty):,}  ({100*(sample-len(nonempty))/sample:.2f}%)")
    cnt = Counter(nonempty)
    print(f"  distinct policy moves:  {len(cnt):,}")
    print(f"  top 15 most-frequent:")
    for mv, c in cnt.most_common(15):
        print(f"    {mv:<8s} {c:>7,}  ({100*c/len(nonempty):5.2f}%)")
    # fraction of buffer made up by top-N moves
    sorted_counts = np.array(sorted(cnt.values(), reverse=True), dtype=np.float64)
    cum = np.cumsum(sorted_counts) / sorted_counts.sum()
    for k in (10, 50, 100, 500):
        if len(cum) >= k:
            print(f"  top-{k:<3d} moves cover {100*cum[k-1]:5.2f}% of policy targets")

    print()
    print("=" * 70)
    print("HISTORY (AlphaZero 8-board stack)")
    print("=" * 70)
    print(f"  records with any history:      {100*(n_filled_slots > 0).mean():.2f}%")
    print(f"  records with all 7 slots full: {100*(n_filled_slots == 7).mean():.2f}%")
    print(f"  mean filled past slots:        {n_filled_slots.mean():.3f} / 7")
    print(f"  records with any past-rep=1:   {100*past_rep_any.mean():.2f}%")
    print(f"  current_repetition fraction:   {100*cur_rep.mean():.2f}%")

    print()
    print("=" * 70)
    print("PLIES REMAINING")
    print("=" * 70)
    print(f"  raw plies_remaining:   min={plies.min():.1f}  median={np.median(plies):.1f}  "
          f"mean={plies.mean():.1f}  max={plies.max():.1f}")
    print(f"  backed plies:          min={bplies.min():.1f}  median={np.median(bplies):.1f}  "
          f"mean={bplies.mean():.1f}  max={bplies.max():.1f}")
    print(f"  distribution of plies_remaining (raw):")
    for lo, hi in [(0,5),(5,20),(20,40),(40,80),(80,160),(160,1000)]:
        f = ((plies>=lo)&(plies<hi)).mean()
        print(f"    [{lo:>3d},{hi:>3d}):  {100*f:5.2f}%")

    print()
    print("=" * 70)
    print("SIDE TO MOVE")
    print("=" * 70)
    print(f"  white-to-move rows: {100*fen_stm.mean():.2f}%   black-to-move: {100*(1-fen_stm.mean()):.2f}%")


if __name__ == "__main__":
    main()
