import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from model import GPT, GPTConfig
from data.parse import dir_iterator
from tqdm import tqdm
import wandb
import os
import json

# Load config from file
with open("configs/config.json", "r") as f:
    config_data = json.load(f)
config = GPTConfig()
scaler = GradScaler()  # Initialize GradScaler for mixed precision

dir_path = config_data["data_path"]
use_wandb = config_data.get("use_wandb", True)
initial_lr = config_data.get("learning_rate", 4e-5)
gradient_accumulation_steps = config_data.get("grad_accum_steps", 5)
num_steps = config_data.get("num_steps", 15000) * gradient_accumulation_steps
config.vocab_size = config_data.get("vocab_size", 1929)
config.block_size = config_data.get("block_size", 256)
model = GPT(config).to("cuda")
opt = torch.optim.Adam(model.parameters(), lr=initial_lr)

# Learning rate scheduler with warmup and cosine decay
def lr_lambda(current_step):
    if current_step < 1000:
        return current_step / 1000
    return 0.5 * (1 + torch.cos(torch.tensor((current_step - 1000) / (num_steps - 1000) * 3.141592653589793))) * 0.1 + 0.9

scheduler = LambdaLR(opt, lr_lambda)

gen = dir_iterator(dir_path,triple = True)
if use_wandb:
    wandb.init(project="ChessRL-pretrain")
    wandb.watch(model)

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
        scheduler.step()
        
        # Calculate accuracy (where argmax(out) == targets and targets != 1928)
        acc = (torch.argmax(out, dim=-1) == targets).float()
        acc = acc[targets != 1928].mean()
        

        progress_bar.set_description(f"Loss: {accumulated_loss:.4f} Accuracy: {acc.item()}")
        if use_wandb:
            wandb.log({"loss": accumulated_loss, "accuracy": acc.item(), "lr": scheduler.get_last_lr()[0]})
        accumulated_loss = 0.0
    
    # Save model checkpoint every 10000 steps
    if (i + 1) % 10000 == 0:
        checkpoint_path = f"pretrain/checkpoint_step_{i+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at step {i+1}")
