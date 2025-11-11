#!/usr/bin/env python3
"""
Training script for chess model with GRPO-like reinforcement loss and WDL value prediction.

This script implements:
1. GRPO Loss: Reinforcement learning loss that scales policy logits by rewards
2. WDL Cross-Entropy Loss: Standard cross-entropy on value_head
3. WDL MSE Loss: Mean squared error on value_head_q probabilities
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.small.model import BT4
from src.train.dataloader import create_dataloader
from src.train.config import TrainingConfig, get_config


class Trainer:
    """Trainer class for chess model with GRPO and WDL losses."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model and load pretrained weights
        self.model = BT4().to(self.device)
        if os.path.exists(config.model_path):
            print(f"Loading model weights from: {config.model_path}")
            checkpoint = torch.load(config.model_path, map_location=self.device, weights_only=False)
            # Handle both raw state dict and checkpoint dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("Model weights loaded successfully")
        else:
            print(f"Warning: Model weights not found at {config.model_path}, using random initialization")
        
        # Initialize frozen base model for KL divergence
        self.base_model = BT4().to(self.device)
        self.base_model.load_state_dict(self.model.state_dict())  # Copy current weights
        self.base_model.eval()  # Set to evaluation mode
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("Frozen base model initialized for KL divergence")
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        
        # Initialize dataloader
        self.dataloader = create_dataloader(
            parquet_file_path=config.data_path,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            chunk_size=config.chunk_size,
            num_workers=config.num_workers,
        )
        
        # Initialize logging
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize wandb
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                entity=config.wandb_entity,
                config=vars(config),
                tags=config.wandb_tags,
                resume="allow" if config.resume_from_checkpoint else None,
            )
            # Watch model gradients and parameters
            # wandb.watch(self.model, log="all", log_freq=config.log_interval)

        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Resume from checkpoint if specified
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)
    
    def compute_grpo_loss(
        self, 
        policy_logits: torch.Tensor, 
        base_policy_logits: torch.Tensor,
        policy_indices: torch.Tensor, 
        rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GRPO (Group Relative Policy Optimization) loss with KL divergence regularization.
        
        The goal is to:
        - Increase logits for moves that led to wins (R=1)
        - Decrease logits for moves that led to losses (R=-1)
        - Prevent policy from diverging too much from base policy (KL divergence)
        
        Formula: -sum(policy_prob[i, policy[i]] * R[i]) / sum(base_policy_prob[i, policy[i]].detach())
        
        Args:
            policy_logits: Policy logits of shape (batch, seq_len, 1929) from trained model
            base_policy_logits: Policy logits of shape (batch, seq_len, 1929) from frozen base model
            policy_indices: Move indices of shape (batch,)
            rewards: Reward values (side_to_move_won) of shape (batch,) with values in {-1, 1}
            
        Returns:
            Tuple of (grpo_loss, kl_divergence)
        """
        batch_size = policy_logits.size(0)
        
        # Clip logits to prevent numerical instability
        # This prevents extreme values that cause inf/nan in softmax
        max_logit = self.config.max_logit_clip
        policy_logits_clipped = torch.clamp(policy_logits[:, 0, :], -max_logit, max_logit)
        base_policy_logits_clipped = torch.clamp(base_policy_logits[:, 0, :], -max_logit, max_logit)
        
        # Compute KL divergence between current policy and base policy
        # KL(base || current) to prevent current policy from diverging too much
        # Convert logits to log probabilities
        log_probs = F.log_softmax(policy_logits_clipped, dim=-1)  # (batch, 1929)
        policy_probs = log_probs.exp()
        base_probs = F.softmax(base_policy_logits_clipped.detach(), dim=-1)  # (batch, 1929)
        
        # Extract probabilities for the played moves (use clipped version for stability)
        selected_probs = policy_probs[torch.arange(batch_size), policy_indices]
        base_selected_probs = base_probs[torch.arange(batch_size), policy_indices]
        
        # Compute GRPO loss using probabilities
        # Multiply selected probabilities by rewards and normalize by sum of base probabilities
        # R=+1 (won) -> increase probability mass
        # R=-1 (lost) -> decrease probability mass
        numerator = torch.sum(selected_probs * rewards)
        denominator = torch.sum(base_selected_probs.detach())
        
        # Negative because we want to maximize (gradient descent minimizes)
        grpo_loss = -numerator / (denominator + 1e-8)  # Add epsilon for numerical stability
        
        # KL divergence: sum(base_probs * (log(base_probs) - log_probs))
        kl_div = F.kl_div(log_probs, base_probs, reduction='batchmean', log_target=False)
        
        # Clip KL divergence to prevent inf/nan propagation
        # If KL is too large, it means the policy has diverged significantly
        max_kl = self.config.max_kl_clip
        kl_div = torch.clamp(kl_div, 0.0, max_kl)
        
        # Check for nan/inf and replace with safe values
        if torch.isnan(grpo_loss) or torch.isinf(grpo_loss):
            grpo_loss = torch.tensor(0.0, device=grpo_loss.device, dtype=grpo_loss.dtype)
        if torch.isnan(kl_div) or torch.isinf(kl_div):
            kl_div = torch.tensor(max_kl, device=kl_div.device, dtype=kl_div.dtype)
        
        return grpo_loss, kl_div
    
    def compute_wdl_losses(
        self, 
        value_h: torch.Tensor, 
        value_h_q: torch.Tensor, 
        wdl_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute WDL (Win/Draw/Loss) losses for both value heads and accuracy.
        
        Args:
            value_h: Value head logits of shape (batch, seq_len, 3)
            value_h_q: Value head Q logits of shape (batch, seq_len, 3)
            wdl_labels: WDL labels of shape (batch,) with values 0/1/2
            
        Returns:
            Tuple of (cross_entropy_loss, mse_loss, wdl_accuracy)
        """
        batch_size = value_h.size(0)
        
        # Reshape to (batch, 3) by selecting seq_len=0
        value_h_flat = value_h[:, 0, :]  # (batch, 3)
        value_h_q_flat = value_h_q[:, 0, :]  # (batch, 3)
        
        # Cross-entropy loss for value_head
        ce_loss = F.cross_entropy(value_h_flat, wdl_labels)
        
        # MSE loss for value_head_q
        # First convert to probabilities
        value_h_q_probs = F.softmax(value_h_q_flat, dim=-1)
        # Convert labels to one-hot
        wdl_one_hot = F.one_hot(wdl_labels, num_classes=3).float()
        # Compute MSE
        mse_loss = F.mse_loss(value_h_q_probs, wdl_one_hot)
        
        # Compute WDL accuracy
        # Get predicted class (argmax of logits)
        wdl_predictions = torch.argmax(value_h_flat, dim=-1)
        # Compute accuracy
        correct_predictions = (wdl_predictions == wdl_labels).sum().item()
        wdl_accuracy = correct_predictions / batch_size
        
        return ce_loss, mse_loss, wdl_accuracy
    
    def train_step(
        self, 
        X: torch.Tensor, 
        policy: torch.Tensor, 
        R: torch.Tensor, 
        WDL: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            X: Board encodings of shape (batch, 8, 8, 19)
            policy: Policy indices of shape (batch,)
            R: Rewards (side_to_move_won) of shape (batch,)
            WDL: WDL labels of shape (batch,)
            
        Returns:
            Dictionary of losses
        """
        # Reshape input: (batch, 8, 8, 19) -> (batch, 1, 8, 8, 19)
        X = X.unsqueeze(1).to(self.device)
        policy = policy.to(self.device)
        R = R.to(self.device)
        WDL = WDL.to(self.device)
        
        # Use automatic mixed precision if enabled
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            # Single forward pass through model to get all outputs
            b, seq_len, _, _, emb = X.size()
            x = X.view(b * seq_len, 64, emb)
            x = self.model.linear1(x)
            x = torch.nn.GELU()(x)
            x = self.model.layernorm1(x)
            x = self.model.ma_gating(x)
            pos_enc = self.model.positional(x)
            
            for i in range(self.model.num_layers):
                x = self.model.layers[i](x, pos_enc)
            
            # Get value heads
            value_h = self.model.value_head(x)
            value_h = value_h.view(b, seq_len, 3)
            value_h_q = self.model.value_head_q(x)
            value_h_q = value_h_q.view(b, seq_len, 3)
            
            # Get policy
            policy_tokens = self.model.policy_tokens_lin(x)
            policy_tokens = torch.nn.GELU()(policy_tokens)
            policy_tokens = policy_tokens + pos_enc
            queries = self.model.queries_pol(policy_tokens)
            keys = self.model.keys_pol(policy_tokens)
            matmul_qk = torch.matmul(queries, torch.transpose(keys, -2, -1))
            dk = torch.sqrt(torch.tensor(self.model.d_model, dtype=matmul_qk.dtype, device=matmul_qk.device))
            policy_attn_logits = matmul_qk / dk
            policy_attn_logits = policy_attn_logits.view(b, seq_len, 64 * 64)
            policy_logits = self.model.policy_head(policy_attn_logits)
            
            # Get base model policy for KL divergence (no gradient)
            with torch.no_grad():
                x_base = X.view(b * seq_len, 64, emb)
                x_base = self.base_model.linear1(x_base)
                x_base = torch.nn.GELU()(x_base)
                x_base = self.base_model.layernorm1(x_base)
                x_base = self.base_model.ma_gating(x_base)
                pos_enc_base = self.base_model.positional(x_base)
                
                for i in range(self.base_model.num_layers):
                    x_base = self.base_model.layers[i](x_base, pos_enc_base)
                
                policy_tokens_base = self.base_model.policy_tokens_lin(x_base)
                policy_tokens_base = torch.nn.GELU()(policy_tokens_base)
                policy_tokens_base = policy_tokens_base + pos_enc_base
                queries_base = self.base_model.queries_pol(policy_tokens_base)
                keys_base = self.base_model.keys_pol(policy_tokens_base)
                matmul_qk_base = torch.matmul(queries_base, torch.transpose(keys_base, -2, -1))
                dk_base = torch.sqrt(torch.tensor(self.base_model.d_model, dtype=matmul_qk_base.dtype, device=matmul_qk_base.device))
                policy_attn_logits_base = matmul_qk_base / dk_base
                policy_attn_logits_base = policy_attn_logits_base.view(b, seq_len, 64 * 64)
                base_policy_logits = self.base_model.policy_head(policy_attn_logits_base)
            
            # Compute losses
            grpo_loss, kl_div = self.compute_grpo_loss(policy_logits, base_policy_logits, policy, R)
            wdl_ce_loss, wdl_mse_loss, wdl_accuracy = self.compute_wdl_losses(value_h, value_h_q, WDL)
            
            # Policy entropy (mean over batch)
            policy_logits_flat = policy_logits[:, 0, :]
            log_probs = F.log_softmax(policy_logits_flat, dim=-1)
            probs = log_probs.exp()
            policy_entropy = (-(probs * log_probs).sum(dim=-1)).mean()

            # Mean of per-sample max/min logits (seq index 0)
            max_per_sample = policy_logits_flat.max(dim=-1).values
            min_per_sample = policy_logits_flat.min(dim=-1).values
            mean_max_logit = max_per_sample.mean()
            mean_min_logit = min_per_sample.mean()
            
            # Weighted total loss
            total_loss = (
                self.config.grpo_loss_weight * grpo_loss +
                self.config.wdl_ce_loss_weight * wdl_ce_loss +
                self.config.wdl_mse_loss_weight * wdl_mse_loss +
                self.config.kl_loss_weight * kl_div
            )
        
        # Backward pass with mixed precision support
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            # Gradient clipping
            if self.config.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            # Gradient clipping
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            self.optimizer.step()
        
        # Detach and convert losses to Python scalars to free memory
        loss_dict = {
            'total_loss': total_loss.detach().item(),
            'grpo_loss': grpo_loss.detach().item(),
            'wdl_ce_loss': wdl_ce_loss.detach().item(),
            'wdl_mse_loss': wdl_mse_loss.detach().item(),
            'kl_div': kl_div.detach().item(),
            'wdl_accuracy': wdl_accuracy,
            'policy_entropy': policy_entropy.detach().item(),
            'mean_max_logit': mean_max_logit.detach().item(),
            'mean_min_logit': mean_min_logit.detach().item(),
        }
        
        # Free memory explicitly
        del total_loss, grpo_loss, wdl_ce_loss, wdl_mse_loss, kl_div
        del policy_logits, base_policy_logits, value_h, value_h_q, x, policy_tokens
        
        return loss_dict
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'grpo_loss': 0.0,
            'wdl_ce_loss': 0.0,
            'wdl_mse_loss': 0.0,
            'kl_div': 0.0,
            'wdl_accuracy': 0.0,
            'policy_entropy': 0.0,
            'mean_max_logit': 0.0,
            'mean_min_logit': 0.0,
        }
        num_batches = 0
        
        for batch_idx, (X, policy, R, WDL) in enumerate(self.dataloader):
            # Train step
            losses = self.train_step(X, policy, R, WDL)
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            num_batches += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                log_str = f"Epoch {self.epoch} | Step {self.global_step} | "
                log_str += " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                print(log_str)
                
                # Wandb logging
                if self.use_wandb:
                    wandb_log = {f"train/{key}": value for key, value in losses.items()}
                    wandb_log["epoch"] = self.epoch
                    wandb_log["global_step"] = self.global_step
                    wandb.log(wandb_log, step=self.global_step)
            
            # Checkpointing
            if self.global_step % self.config.checkpoint_interval == 0 and self.global_step > 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
            
            self.global_step += 1
            
            # Check if we've reached max_steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self):
        """Main training loop."""
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        # Print full configuration
        print("\nTraining Configuration:")
        print("-" * 80)
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.get_effective_batch_size()}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Weight decay: {self.config.weight_decay}")
        print(f"  Max gradient norm: {self.config.max_grad_norm}")
        print(f"  Mixed precision: {self.config.mixed_precision}")
        print(f"\n  Loss weights:")
        print(f"    - GRPO: {self.config.grpo_loss_weight}")
        print(f"    - WDL CE: {self.config.wdl_ce_loss_weight}")
        print(f"    - WDL MSE: {self.config.wdl_mse_loss_weight}")
        print(f"    - KL Divergence: {self.config.kl_loss_weight}")
        print(f"\n  Numerical stability:")
        print(f"    - Max logit clip: {self.config.max_logit_clip}")
        print(f"    - Max KL clip: {self.config.max_kl_clip}")
        print(f"\n  Training:")
        print(f"    - Epochs: {self.config.num_epochs}")
        print(f"    - Max steps: {self.config.max_steps if self.config.max_steps else 'None (use epochs)'}")
        print(f"    - Log interval: {self.config.log_interval}")
        print(f"    - Checkpoint interval: {self.config.checkpoint_interval}")
        print(f"\n  Data:")
        print(f"    - Data path: {self.config.data_path}")
        print(f"    - Chunk size: {self.config.chunk_size}")
        print(f"    - Shuffle: {self.config.shuffle}")
        print(f"\n  Paths:")
        print(f"    - Checkpoint dir: {self.config.checkpoint_dir}")
        print(f"\n  Logging:")
        print(f"    - Wandb: {self.use_wandb}")
        if self.use_wandb:
            print(f"    - Project: {self.config.wandb_project}")
            if self.config.wandb_run_name:
                print(f"    - Run name: {self.config.wandb_run_name}")
            if self.config.wandb_entity:
                print(f"    - Entity: {self.config.wandb_entity}")
        
        # Print model size
        print("\nModel Information:")
        print("-" * 80)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 80)
            
            # Train one epoch
            epoch_losses = self.train_epoch()
            
            # Log epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            for key, value in epoch_losses.items():
                print(f"  {key}: {value:.4f}")
            
            # Wandb epoch summary
            if self.use_wandb:
                wandb_epoch_log = {f"epoch/{key}": value for key, value in epoch_losses.items()}
                wandb_epoch_log["epoch"] = epoch
                wandb.log(wandb_epoch_log, step=self.global_step)
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Check if we've reached max_steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                print(f"\nReached max_steps={self.config.max_steps}. Stopping training.")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Final step: {self.global_step}")
        print("=" * 80)
        
        # Save final checkpoint
        self.save_checkpoint("checkpoint_final.pt")
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, filename: str):
        """
        Save a training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train chess model with GRPO and WDL losses'
    )
    
    # Configuration overrides
    parser.add_argument('--data-path', type=str, default='data/shuffled_data.parquet',
                        help='Path to training data parquet file')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size (default: 1024)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum number of steps (overrides num_epochs if set)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    
    # Loss weights
    parser.add_argument('--grpo-weight', type=float, default=1.0,
                        help='GRPO loss weight (default: 1.0)')
    parser.add_argument('--wdl-ce-weight', type=float, default=0.05,
                        help='WDL cross-entropy loss weight (default: 0.05)')
    parser.add_argument('--wdl-mse-weight', type=float, default=0.05,
                        help='WDL MSE loss weight (default: 0.05)')
    parser.add_argument('--kl-weight', type=float, default=1,
                        help='KL divergence loss weight (default: 0.1)')
    
    # Numerical stability
    parser.add_argument('--max-logit-clip', type=float, default=10.0,
                        help='Maximum logit value for clipping (default: 10.0)')
    parser.add_argument('--max-kl-clip', type=float, default=10.0,
                        help='Maximum KL divergence value for clipping (default: 10.0)')
    
    # Wandb settings
    parser.add_argument('--use-wandb', action='store_true', default=True,
                        help='Use Weights & Biases logging (default: True)')
    parser.add_argument('--no-wandb', action='store_false', dest='use_wandb',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='chess-rl-training',
                        help='Wandb project name (default: chess-rl-training)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity/username (default: None)')
    
    # Memory optimization
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Enable mixed precision (FP16) training (default: False)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = get_config(
        data_path=args.data_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        resume_from_checkpoint=args.resume,
        grpo_loss_weight=args.grpo_weight,
        wdl_ce_loss_weight=args.wdl_ce_weight,
        wdl_mse_loss_weight=args.wdl_mse_weight,
        kl_loss_weight=args.kl_weight,
        max_logit_clip=args.max_logit_clip,
        max_kl_clip=args.max_kl_clip,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        mixed_precision=args.mixed_precision,
    )
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

