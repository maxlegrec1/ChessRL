import torch
from torch.amp import autocast, GradScaler
from model import GPT, GPTConfig
from fine_tune import gen
from tqdm import tqdm
import wandb
import os
from data.vocab import policy_index
config = GPTConfig()
config.vocab_size = 1929
config.block_size = 256
model = GPT(config).to("cuda")
model.load_state_dict(torch.load("checkpoint_step_15000.pt"))
scaler = GradScaler()  # Initialize GradScaler for mixed precision

  
start_think_index = policy_index.index("<thinking>")
end_think_index = policy_index.index("</thinking>")
end_variation_index = policy_index.index("end_variation")
end_index = policy_index.index("end")


dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
num_steps = 15000
gen = gen()
use_wandb = True
if use_wandb:
    wandb.init(project="ChessRL-fine-tune")
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
    start_think_acc = acc[targets == start_think_index].mean()
    end_think_acc = acc[targets == end_think_index].mean()
    end_variation_acc = acc[targets == end_variation_index].mean()
    end_acc = acc[targets == end_index].mean()
    acc = acc[targets != 1928].mean()
    progress_bar.set_description(f"Loss: {loss.item()} Accuracy: {acc.item()}")
    if use_wandb:
        wandb.log({"loss": loss.item(), "accuracy": acc.item(), "start_think_acc": start_think_acc.item(), "end_think_acc": end_think_acc.item(), "end_variation_acc": end_variation_acc.item(), "end_acc": end_acc.item()})
    
    # Save model checkpoint every 1000 steps
    if (i + 1) % 1000 == 0:
        checkpoint_path = f"checkpoint_step_fine_tune_{i+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at step {i+1}")
    if i == 700:
        checkpoint_path = f"checkpoint_step_fine_tune_{i+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)