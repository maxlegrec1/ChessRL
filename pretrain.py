import torch
from torch.amp import autocast, GradScaler
from model import GPT, GPTConfig
from data.parse import dir_iterator
from tqdm import tqdm
import wandb
import os

config = GPTConfig()
config.vocab_size = 1929
config.block_size = 256
config.n_layer = 12
config.n_head = 16
config.n_embd = 2048
model = GPT(config).to("cuda")

scaler = GradScaler()  # Initialize GradScaler for mixed precision

dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
opt = torch.optim.Adam(model.parameters(), lr=4e-5)

gen = dir_iterator(dir_path)
use_wandb = True
if use_wandb:
    wandb.init(project="ChessRL-pretrain")
    wandb.watch(model)

gradient_accumulation_steps = 64  # Number of steps to accumulate gradients
num_steps = 15000*gradient_accumulation_steps
progress_bar = tqdm(range(num_steps))
accumulated_loss = 0.0

for i in progress_bar:
    inp = next(gen)
    
    with autocast(device_type='cuda', dtype=torch.float16):  # Enable mixed precision
        out, loss, targets = model(inp, compute_loss=True)
    
    loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
    scaler.scale(loss).backward()
    accumulated_loss += loss.item()
    
    if (i + 1) % gradient_accumulation_steps == 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        
        # Calculate accuracy (where argmax(out) == targets and targets != 1928)
        acc = (torch.argmax(out, dim=-1) == targets).float()
        acc = acc[targets != 1928].mean()
        
        progress_bar.set_description(f"Loss: {accumulated_loss:.4f} Accuracy: {acc.item()}")
        if use_wandb:
            wandb.log({"loss": accumulated_loss, "accuracy": acc.item()})
        accumulated_loss = 0.0
    
    # Save model checkpoint every 1000 steps
    if (i + 1) % 1000 == 0:
        checkpoint_path = f"pretrain/checkpoint_step_{i+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at step {i+1}")
