# Model Quality Analysis

## Date: December 3, 2025

This document summarizes findings from analyzing the small model's quality on tactical positions, specifically mate-in-one positions.

---

## 1. Bug Fix: Q-Value Sign Issue

### Problem
MCTS was returning Q-values with the **wrong sign** for moves. A move leading to a bad position for the player showed positive Q-values instead of negative.

### Root Cause
Python was loading a **stale compiled library** (`cpp/_inference_cpp.cpython-313-x86_64-linux-gnu.so` from Nov 26) instead of the freshly built version (Dec 2) that included the Q-value negation fix.

The issue was in `src/mcts/__init__.py` which added `cpp/` to the front of `sys.path`:
```python
_CPP_DIR = _ROOT / "cpp"
if str(_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_DIR))  # This caused stale .so to be loaded
```

### Fix
1. Removed the `sys.path` hack from `src/mcts/__init__.py`
2. Updated `build.sh` to clean stale `.so` files:
```bash
rm -f ./cpp/*.so  # Clean up any stale .so files
```

---

## 2. Mate-in-One Position Analysis

### Test Case
- **Root position**: `r2r4/pp1k2pp/2pP4/Q4pN1/2P3P1/1B5b/PP1R3P/RNB1K3 b Q - 2 27` (black to move)
- **Move d7c8**: King moves from d7 to c8
- **Resulting position**: `r1kr4/pp4pp/2pP4/Q4pN1/2P3P1/1B5b/PP1R3P/RNB1K3 w Q - 3 28` (white to move)
- **White has Qc7# (checkmate in 1)**

### Model Evaluation of Position After d7c8

| Metric | Expected | Model Output |
|--------|----------|--------------|
| Value (WDL) | [1.0, 0.0, 0.0] | [0.38, 0.32, 0.30] |
| Best move | Qc7# (a5c7) | Qa4 (a5a4) |

The model correctly identifies white as slightly better (+8%) but completely fails to recognize the forced checkmate.

### Policy Head Analysis: Qc7# Probability

| Checkpoint | Qc7# Rank | Qc7# Probability |
|------------|-----------|------------------|
| initial_teacher.pt | #10 | 2.54% |
| step_00000100.pt | #28 | 0.43% |
| step_00002800.pt | #28 | 0.62% |

**Key Finding**: The untrained model gave Qc7# higher probability (2.54%) than trained models because it has uniform distribution. Training taught the model to prefer quiet moves over tactical moves.

---

## 3. MCTS Exploration Issue

### Why Checkmate Isn't Found
When MCTS runs from the root with 800 simulations:
- d7c8 gets ~72 visits (9% of total)
- From d7c8 position, white has ~40 legal moves
- Qc7# has only 0.62% prior probability
- With ~72 visits distributed among 40 moves, Qc7 gets ~0-2 visits
- Not enough exploration to dominate the Q-value average

When running MCTS directly from the position after d7c8 with 800 simulations:
- Qc7# gets 505 visits and Q = +1.0 (checkmate correctly found!)

**Conclusion**: The policy head's low probability for Qc7 prevents MCTS from exploring the checkmate line sufficiently.

---

## 4. Training Progression Analysis

### Mating Move Probability Over 100 Mate-in-One Positions

| Checkpoint | Avg Prob | Avg Rank | Median Prob | Median Rank |
|------------|----------|----------|-------------|-------------|
| initial_teacher | 2.56% | 22.7 | 2.46% | 24.5 |
| step_100 | 1.76% | 25.7 | 1.18% | 27.0 |
| step_2800 | 2.99% | 21.0 | 1.88% | 17.0 |
| step_3600 | **3.29%** | **19.8** | 1.90% | 19.0 |

### Value Accuracy on Mate-in-One Positions

| Checkpoint | Value (W > L) | Value (W > D & W > L) | Policy Correct |
|------------|---------------|----------------------|----------------|
| step_100 | 71% | 2% | 3% |
| step_2800 | 84% | 57% | 5% |

### Key Observations

1. **"Worse Before Better" Pattern**: At step 100, the model performs worse than random (uniform) because it has learned to prefer quiet moves but hasn't yet learned which quiet moves are good.

2. **Steady Improvement**: From step 100 → 2800 → 3600, all metrics improve:
   - Average mating move probability: 1.76% → 2.99% → 3.29%
   - Average rank: 25.7 → 21.0 → 19.8
   - Value accuracy: 71% → 84%

3. **Policy Still Weak**: Even at step 3600, only 3.29% average probability for mating moves (should be >50% ideally).

---

## 5. Critical Bug: Broken mirror_fen() Function

### Discovery
Analysis of value accuracy on mate-in-one positions revealed a significant **side-to-move bias**:

| Side to Move | Accuracy (W > L) | True Accuracy (W > D > L) |
|--------------|------------------|---------------------------|
| White | 64.7% | 43.1% |
| Black | **95.9%** | **81.6%** |

When positions were mirrored (swapping colors), the accuracy **followed the side**, not the position:
- Original white positions (now black): 92.2% accuracy
- Original black positions (now white): 67.4% accuracy

### Root Cause
The `mirror_fen()` function in `src/train/dataset.py` had a critical bug:

```python
# BUGGY CODE
def mirror_fen(fen: str) -> str:
    board = chess.Board(fen)
    board.mirror()  # Returns NEW board, but result is DISCARDED!
    return board.fen()  # Returns ORIGINAL FEN, not mirrored!
```

`chess.Board.mirror()` returns a **new board** and does NOT modify in-place. The result was being discarded.

### Impact on Training
During training, 50% of positions are "mirrored" for data augmentation. The bug caused:
- **FEN**: Original (unmirored) 
- **Policy**: Mirrored (via `mirror_policy_str`)
- **WDL**: Original (correct for actual mirroring)

This created **corrupted training data** where the policy target pointed to completely wrong squares (rank 1 ↔ rank 8 swapped for policy but not the board).

### Fix
```python
# FIXED CODE
def mirror_fen(fen: str) -> str:
    board = chess.Board(fen)
    mirrored = board.mirror()  # Capture the returned board
    return mirrored.fen()
```

### Retrain Required
All models trained with the buggy code have learned incorrect policy associations. **Retraining is required** after this fix.

---

## 6. Recommendations

1. **Retrain Immediately**: The `mirror_fen()` bug corrupted 50% of training data. All models need retraining.

2. **More Training**: The model is improving but needs more steps to learn tactical patterns.

3. **Training Data**: Include more tactical positions in self-play data. Consider:
   - Adding puzzle positions to training
   - Weighting tactical positions higher in loss
   - Using Stockfish to identify forcing moves and boost their targets

4. **Architecture**: Consider adding auxiliary heads for:
   - Check detection
   - Threat detection
   - Material evaluation

5. **Search Parameters**: With weak policy, higher `cpuct` values might help explore more moves, but this is a band-aid not a fix.

---

## 7. Critical Bug: MCTS Temperature Setting

### Discovery
Even after retraining with the fixed `mirror_fen()`, the trained model performed poorly in brawls despite better metrics:
- Without MCTS: trained model scores 514.5 - 485.5 vs random (slightly better)
- With MCTS: trained model loses 28 - 74 to random (much worse!)

Investigation revealed that MCTS with the trained model wasn't exploring checkmates, even with 800 simulations.

### Root Cause
The `SmallMCTSInference.get_move_from_fen_no_thinking()` method passed `temperature=T` (default 0.1) to the internal MCTS search. This temperature is used for the **policy softmax during tree expansion**, not just for final move selection.

With `temperature=0.1`:
- The policy becomes extremely peaked
- Only 1-6 moves get explored (most have Q=0.0)
- Checkmates are completely missed even with 800 simulations

With `temperature=1.0`:
- Policy priors allow proper exploration
- 17-27 moves get explored
- Checkmates are found correctly (Q=1.0)

### Fix
Modified `src/models/small/inference_mcts.py` to use:
- `temperature=1.0` for internal MCTS search (enables exploration)
- `temperature_at_root=T` for final action selection (controls move determinism)

```python
res = self.mcts.run(
    [fen],
    temperature=1.0,  # Internal search - keep at 1.0 for exploration
    simulations=self.simulations,
    cpuct=self.cpuct,
    temperature_at_root=T,  # Final action selection
    use_dirichlet=False
)
```

---

## 8. Summary

The model is learning but has significant weaknesses in tactical positions:
- Value head correctly identifies winning side but underestimates winning margins
- Policy head fails to prioritize forcing moves (checks, captures, threats)
- Training is progressing in the right direction but slowly

The fundamental issue is that self-play generates mostly quiet positions, so the model learns to play quiet moves. Tactical positions are rare in training data, so the model doesn't learn to recognize them.

