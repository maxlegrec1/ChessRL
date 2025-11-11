#!/usr/bin/env python3
"""
Training configuration for chess model.

This module provides a dataclass-based configuration system for easily
adjusting training hyperparameters without modifying the training code.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training the chess model."""
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Batch settings
    batch_size: int = 1024
    gradient_accumulation_steps: int = 1
    
    # Loss weights
    grpo_loss_weight: float = 1.0
    wdl_ce_loss_weight: float = 1.0  # Cross-entropy loss for value_head
    wdl_mse_loss_weight: float = 1.0  # MSE loss for value_head_q
    kl_loss_weight: float = 1  # KL divergence loss weight
    
    # Numerical stability
    max_logit_clip: float = 10.0  # Clip policy logits to prevent numerical instability
    max_kl_clip: float = 10.0  # Clip KL divergence to prevent inf/nan propagation
    
    # Training parameters
    num_epochs: int = 10
    max_steps: Optional[int] = None  # If set, overrides num_epochs
    
    # Logging and checkpointing
    log_interval: int = 1  # Log every N steps
    checkpoint_interval: int = 1000  # Save checkpoint every N steps
    eval_interval: int = 500  # Evaluate every N steps (if eval set available)
    
    # Weights & Biases (wandb) settings
    use_wandb: bool = True  # Enable wandb logging
    wandb_project: str = "chess-rl-training"
    wandb_run_name: Optional[str] = None  # Auto-generated if None
    wandb_entity: Optional[str] = None  # Your wandb username/team
    wandb_tags: list = field(default_factory=list)  # Tags for the run
    
    # Paths
    data_path: str = "data/shuffled_data.parquet"
    checkpoint_dir: str = "checkpoints"
    
    # Model settings
    device: str = "cuda"
    mixed_precision: bool = False  # Use FP16 mixed precision training
    # model_path: str = "src/models/small/model.pt"  # Path to pretrained model weights
    model_path: str = "src/models/small/model.pt"  # Path to pretrained model weights
    # Data loading
    num_workers: int = 0
    chunk_size: int = 10000
    shuffle: bool = True
    
    # Model path (for resuming training)
    resume_from_checkpoint: Optional[str] = None
    
    # Gradient clipping
    max_grad_norm: Optional[float] = 1.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
    
    def get_effective_batch_size(self) -> int:
        """Get the effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


# Default configuration instance
default_config = TrainingConfig()


def get_config(**kwargs) -> TrainingConfig:
    """
    Get a training configuration with optional overrides.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        TrainingConfig instance with specified overrides
        
    Example:
        >>> config = get_config(learning_rate=5e-5, batch_size=512)
    """
    return TrainingConfig(**kwargs)

