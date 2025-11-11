# Chess Model Training

This directory contains the training infrastructure for the chess model with GRPO-like reinforcement loss and WDL value prediction.

## Overview

The training system implements:
1. **GRPO Loss**: Group Relative Policy Optimization - a reinforcement learning loss that scales policy logits by game outcomes
2. **WDL Cross-Entropy Loss**: Standard classification loss on the value head for Win/Draw/Loss prediction
3. **WDL MSE Loss**: Mean squared error on the value_head_q probabilities
4. **KL Divergence Loss**: Regularization to prevent the trained policy from diverging too much from the original pretrained policy

## Files

- `config.py`: Configuration dataclass with all training hyperparameters
- `train.py`: Main training script with loss computation and training loop
- `dataloader.py`: PyTorch DataLoader for chess training data from parquet files

## Quick Start

### Basic Training

```bash
python src/train/train.py --data-path data/shuffled_data.parquet
```

### Custom Hyperparameters

```bash
python src/train/train.py \
    --data-path data/shuffled_data.parquet \
    --batch-size 512 \
    --learning-rate 5e-5 \
    --num-epochs 20 \
    --checkpoint-dir my_checkpoints \
    --log-dir my_logs
```

### Resume Training

```bash
python src/train/train.py \
    --data-path data/shuffled_data.parquet \
    --resume checkpoints/checkpoint_epoch_5.pt
```

### Adjust Loss Weights

```bash
python src/train/train.py \
    --data-path data/shuffled_data.parquet \
    --grpo-weight 2.0 \
    --wdl-ce-weight 1.0 \
    --wdl-mse-weight 0.5
```

## Configuration

### Default Hyperparameters

- **Learning Rate**: 1e-4
- **Batch Size**: 128
- **Optimizer**: Adam (β1=0.9, β2=0.999)
- **Weight Decay**: 1e-5
- **Gradient Clipping**: Max norm = 1.0
- **Pretrained Model**: Loaded from `src/models/small/model.pt`

### Loss Weights

Loss weights can be adjusted independently:
- `grpo_loss_weight`: Weight for GRPO reinforcement loss (default: 1.0)
- `wdl_ce_loss_weight`: Weight for WDL cross-entropy loss (default: 1.0)
- `wdl_mse_loss_weight`: Weight for WDL MSE loss (default: 1.0)
- `kl_loss_weight`: Weight for KL divergence regularization (default: 0.1)

### Modifying Configuration Programmatically

```python
from src.train.config import get_config
from src.train.train import Trainer

# Create custom configuration
config = get_config(
    learning_rate=5e-5,
    batch_size=512,
    grpo_loss_weight=2.0,
    max_steps=100000,
)

# Train
trainer = Trainer(config)
trainer.train()
```

## Data Format

The dataloader expects a parquet file with the following columns:
- `board_encoding`: Board representation (flattened to 1216 = 8×8×19)
- `policy_index`: UCI move index (0-1928)
- `side_to_move_won`: Reward signal (-1 for loss, 1 for win)
- `wdl`: Win/Draw/Loss label (0=black win, 1=draw, 2=white win)

## Loss Formulas

### GRPO Loss

```
GRPO Loss = -sum(policy_logit[i, policy[i]] * R[i]) / sum(abs(policy_logit[i, policy[i]]).detach())
```

Where:
- `policy_logit[i, policy[i]]` is the logit for the played move
- `R[i]` is the reward with values in {-1, 1}:
  - R=+1 (won) → increase logit for this move
  - R=-1 (lost) → decrease logit for this move
- Denominator uses absolute values for stable normalization
- Denominator is detached to prevent gradient flow through normalization

**Note:** The GRPO loss can be negative, which is normal! A negative loss means the model is performing well on that batch (i.e., assigning high logits to winning moves and low logits to losing moves).

### KL Divergence Loss

```
KL Loss = KL(base_policy || current_policy)
```

Where:
- `base_policy`: Frozen pretrained model's policy distribution
- `current_policy`: Trained model's policy distribution
- Uses `F.kl_div` with `reduction='batchmean'`

**Purpose:** Prevents the trained policy from diverging too far from the original pretrained policy. This acts as a regularizer to maintain stable training and preserve knowledge from the original model.

**Recommended weight:** 0.01 - 0.5 depending on how much deviation you want to allow.

### WDL Cross-Entropy Loss

```
WDL CE Loss = CrossEntropy(value_head_output, WDL_label)
```

Standard classification loss on the value head.

### WDL MSE Loss

```
WDL MSE Loss = MSE(softmax(value_head_q_output), one_hot(WDL_label))
```

Mean squared error between softmax probabilities and one-hot encoded labels.

## Monitoring

Training metrics are logged to **Weights & Biases (wandb)**.

### Weights & Biases

Wandb logging is enabled by default. To use it:

1. Install wandb:
```bash
pip install wandb
```

2. Login to wandb:
```bash
wandb login
```

3. Run training (wandb logging is automatic):
```bash
python src/train/train.py --data-path data/shuffled_data.parquet
```

4. View metrics at: https://wandb.ai/your-username/chess-rl-training

### Disable Wandb

To disable wandb logging:
```bash
python src/train/train.py --no-wandb
```

### Custom Wandb Configuration

```bash
python src/train/train.py \
    --wandb-project my-chess-project \
    --wandb-run-name experiment-v1 \
    --wandb-entity my-team
```

### Logged Metrics

All training runs log the following metrics:
- Total loss (weighted sum of all losses)
- GRPO loss
- WDL cross-entropy loss
- WDL MSE loss
- Epoch number
- Global step
- Model gradients and parameters (wandb only)

## Checkpointing

Checkpoints are saved:
- Every `checkpoint_interval` steps (default: 1000)
- At the end of each epoch
- At the end of training (`checkpoint_final.pt`)

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Current epoch and global step
- Configuration

**Note**: When wandb is enabled, checkpoints are automatically uploaded to Weights & Biases for cloud storage and version control.

## Model Architecture

The training system uses **two models**:

### 1. Trainable Model
The model being optimized during training:
- Loaded from pretrained weights (`src/models/small/model.pt`)
- Parameters updated via gradient descent
- Outputs policy and value predictions

### 2. Frozen Base Model
A copy of the initial pretrained model:
- Identical architecture to trainable model
- Weights frozen (not updated during training)
- Used only for computing KL divergence
- Ensures trained policy doesn't drift too far from original

### Model Flow

Both models process inputs as:
1. Input: `(batch, 8, 8, 19)` board encoding
2. Reshaped to: `(batch, 1, 8, 8, 19)` for sequence dimension
3. Outputs:
   - Policy logits: `(batch, 1, 1929)` - move probabilities
   - Value head: `(batch, 1, 3)` - WDL logits for cross-entropy
   - Value head Q: `(batch, 1, 3)` - WDL logits for MSE

## Command-Line Arguments

```
--data-path          Path to training data parquet file
--batch-size         Batch size (default: 1024)
--learning-rate      Learning rate (default: 1e-4)
--num-epochs         Number of epochs (default: 10)
--max-steps          Maximum number of steps (overrides num_epochs)
--checkpoint-dir     Checkpoint directory (default: checkpoints)
--device             Device to use (default: cuda)
--resume             Resume from checkpoint path
--grpo-weight        GRPO loss weight (default: 1.0)
--wdl-ce-weight      WDL CE loss weight (default: 1.0)
--wdl-mse-weight     WDL MSE loss weight (default: 1.0)
--kl-weight          KL divergence loss weight (default: 0.1)
--use-wandb          Enable wandb logging (default: True)
--no-wandb           Disable wandb logging
--wandb-project      Wandb project name (default: chess-rl-training)
--wandb-run-name     Wandb run name (default: auto-generated)
--wandb-entity       Wandb entity/username (default: None)
--mixed-precision    Enable FP16 mixed precision training (default: False)
```

## Memory Optimization

The training script includes several memory optimization features:

### 1. Single Forward Pass
The training loop performs only **one** forward pass per batch (fixed from previous double-pass issue), significantly reducing memory usage.

### 2. Mixed Precision Training (FP16)
Enable mixed precision to reduce memory usage by ~50%:
```bash
python src/train/train.py --mixed-precision
```

Benefits:
- Halves memory footprint
- Can double batch size
- Faster training on modern GPUs (Volta+)

### 3. Automatic Memory Management
- Explicit tensor deletion after use
- Gradient tensors properly freed


## Troubleshooting

### Out of Memory

Try these solutions in order:

1. **Enable mixed precision** (most effective):
```bash
python src/train/train.py --mixed-precision --batch-size 256
```

2. **Reduce batch size**:
```bash
python src/train/train.py --batch-size 64
```

3. **Use gradient accumulation** (effective batch size without memory):
```python
config = get_config(
    batch_size=64,
    gradient_accumulation_steps=4  # Effective batch size = 256
)
```


### Expected Memory Usage

For a 32M parameter model:
- **FP32**: ~0.5-1 GB (model + optimizer)
- **Batch size 128**: ~2-4 GB total
- **Batch size 256**: ~4-8 GB total
- **With FP16**: ~50% reduction

If you're seeing >10 GB for batch size 128, something is wrong.

### Training Too Slow

The dataloader reads from parquet in chunks. Adjust chunk size:
```python
config = get_config(chunk_size=50000)  # Larger chunks
```

### Loss Not Decreasing

Try adjusting loss weights or learning rate:
```bash
python src/train/train.py --learning-rate 5e-5 --grpo-weight 0.5
```

