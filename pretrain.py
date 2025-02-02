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
model = GPT(config).to("cuda")

scaler = GradScaler()  # Initialize GradScaler for mixed precision

dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
num_steps = 15000
gen = dir_iterator(dir_path)
use_wandb = True
if use_wandb:
    wandb.init(project="ChessRL-pretrain")
    wandb.watch(model)

progress_bar = tqdm(range(num_steps))
for i in progress_bar:
    inp = next(gen)
    
    with autocast(device_type='cuda', dtype=torch.float16):  # Enable mixed precision
        out, loss, targets = model(inp, compute_loss=True)
    
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    
    # Calculate accuracy (where argmax(out) == targets and targets != 1928)
    acc = (torch.argmax(out, dim=-1) == targets).float()
    acc = acc[targets != 1928].mean()
    
    progress_bar.set_description(f"Loss: {loss.item()} Accuracy: {acc.item()}")
    if use_wandb:
        wandb.log({"loss": loss.item(), "accuracy": acc.item()})
    
    # Save model checkpoint every 1000 steps
    if (i + 1) % 1000 == 0:
        checkpoint_path = f"checkpoint_step_{i+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at step {i+1}")
