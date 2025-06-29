import torch
from torch.amp import autocast, GradScaler
from new_paradigm.model_v3_raw import BT4

from utils.parse_paradigm import batch_generator
from metrics.tactics_evaluation_improved import calculate_tactics_accuracy
from metrics.endgame_evaluation import calculate_endgame_score
from metrics.play_move_gemini import main_model_vs_model,main_model_vs_stockfish
from tqdm import tqdm
import wandb
import os
import chess
from utils.vocab import policy_index

model = BT4().to("cuda")

pretrained_dict = torch.load("pretrain/paradigm_400000.pt")
model_dict = model.state_dict()

successfully_loaded_layers = 0
skipped_shape_mismatch = 0
skipped_missing_key = 0

for key, model_tensor in model_dict.items():
    try:
        if key in pretrained_dict:
            pretrained_tensor = pretrained_dict[key]
            if model_tensor.shape == pretrained_tensor.shape:
                model_dict[key] = pretrained_tensor
                print(f"Successfully loaded layer: {key}")
                successfully_loaded_layers += 1
            else:
                print(f"Warning: Layer {key} skipped. Shape mismatch. Model: {model_tensor.shape}, Pretrained: {pretrained_tensor.shape}")
                skipped_shape_mismatch += 1
        else:
            print(f"Warning: Layer {key} not found in pretrained weights. Will use initialized weights.")
            skipped_missing_key += 1
    except Exception as e:
        print(f"Error loading layer {key}: {str(e)}. Will use initialized weights.")
        # Potentially add to a counter for other errors if needed

model.load_state_dict(model_dict, strict=False) # Use strict=False to handle missing/mismatched keys gracefully
print(f"\nSummary of state_dict loading:")
print(f"  Successfully loaded layers: {successfully_loaded_layers}")
print(f"  Skipped due to shape mismatch: {skipped_shape_mismatch}")
print(f"  Skipped due to missing key in pretrained_dict: {skipped_missing_key}")

#print the number of parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
scaler = GradScaler()  # Initialize GradScaler for mixed precision
dir_path = "/media/maxime/385e67e0-8703-4723-a0bc-af3a292fd030/stockfish_data/Leela_training"
gen = batch_generator(dir_path, batch_size=20, return_fen=False, triple=True, device='cuda')

opt = torch.optim.Adam(model.parameters(), lr=1e-4)


use_wandb = True
if use_wandb:
    wandb.init(project="ChessRL-pretrain")
    #wandb.watch(model)

gradient_accumulation_steps = 3  # Number of steps to accumulate gradients
num_steps = 50_000_000*gradient_accumulation_steps
progress_bar = tqdm(range(num_steps))
accumulated_loss = 0.0
step_counter = 0
enable_float16 = True
for i in progress_bar:
    inp = next(gen)
    
    with autocast(device_type='cuda', dtype=torch.float16, enabled= enable_float16):  # Enable mixed precision
        out, loss, targets = model(inp,compute_loss=True)
    loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
    scaler.scale(loss).backward()
    accumulated_loss += loss.item()
    if (i + 1) % gradient_accumulation_steps == 0:
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        # Calculate accuracy (where argmax(out) == targets and targets != 1928)
        acc = (torch.argmax(out, dim=-1) == targets).float()
        acc = acc[targets != 1928].mean()
        fens = inp[2]
        if step_counter % 5000 == 0:
            tactics_accuracy = calculate_tactics_accuracy(model, num_positions=1000, is_thinking_model=False, T=0.1)
            endgame_score = calculate_endgame_score(model, T=0.1, is_thinking_model=False, limit_elo=False, num_positions=100)
            win_rate_vs_stockfish,elo = main_model_vs_stockfish(model=model,model1_name=f"{step_counter}", temp=0.1, num_games=40)
            win_rate_vs_model = main_model_vs_model(model1=model,model1_name=f"{step_counter}", temp=0.4, num_games=100)
        else:
            pass
        step_counter += 1
        progress_bar.set_description(f"Loss: {accumulated_loss:.4f} Accuracy: {acc.item()}")

        if use_wandb:
            log_dict = {
                "loss": accumulated_loss,
                "accuracy": acc.item(),
                "lr": opt.param_groups[0]["lr"],
            }
            log_dict["tactics_accuracy"] = tactics_accuracy
            log_dict["endgame_score"] = endgame_score
            log_dict["win_rate_vs_stockfish"] = win_rate_vs_stockfish
            log_dict["elo"] = elo
            log_dict["win_rate_vs_model"] = win_rate_vs_model
            wandb.log(log_dict)
        accumulated_loss = 0.0
    
    # Save model checkpoint every 1000 steps
    if (i + 1) % 100000 == 0:
        #pass
        checkpoint_path = f"pretrain/v3_{i+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at step {i+1}")
