import torch
from torch.amp import autocast, GradScaler
from new_paradigm.value_q import BT4

from utils.parse_value_leela import leela_data_generator
from metrics.tactics_evaluation_improved import calculate_tactics_accuracy
from metrics.endgame_evaluation import calculate_endgame_score
from metrics.play_move_gemini import main_model_vs_model,main_model_vs_stockfish
from tqdm import tqdm
import wandb
import os
import chess
from utils.vocab import policy_index

model = BT4().to("cuda")
#model.load_state_dict(torch.load("pretrain/value_only_71000.pt"))
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
scaler = GradScaler()  # Initialize GradScaler for mixed precision
config_path = "/mnt/2tb/LeelaDataReader/lczero-training/tf/configs/example.yaml"
gen = leela_data_generator(config_path)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)


use_wandb = True
if use_wandb:
    wandb.init(project="ChessRL-pretrain")
    #wandb.watch(model)

gradient_accumulation_steps = 1  # Number of steps to accumulate gradients
num_steps = 50_000_000*gradient_accumulation_steps
progress_bar = tqdm(range(num_steps))
accumulated_loss = 0.0
accumulated_value_loss = 0.0
accumulated_policy_loss = 0.0
accumulated_q_loss = 0.0
step_counter = 0
epoch = 0
enable_float16 = True
for i in progress_bar:
    #at 10000, change lr to 0 and generator to block size 64
    try:
        inp = next(gen)
    except Exception as e:
        epoch += 1
        del gen
        gen = leela_data_generator(config_path)
        inp = next(gen)
    #print(inp[2],inp[3])
    with autocast(device_type='cuda', dtype=torch.float16, enabled= enable_float16):  # Enable mixed precision
        #print(inp[0].shape,inp[3].shape,inp[4].shape)
        out,value,loss_policy, loss_value, loss_q, targets, true_values = model(inp,compute_loss=True)
        loss = loss_value + loss_q #+ loss_policy
    loss_policy = loss_policy / gradient_accumulation_steps
    loss_value = loss_value / gradient_accumulation_steps
    loss_q = loss_q / gradient_accumulation_steps
    loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
    scaler.scale(loss).backward()
    accumulated_loss += loss.item()
    accumulated_value_loss += loss_value.item()
    accumulated_policy_loss += loss_policy.item()
    accumulated_q_loss += loss_q.item()
    if (i + 1) % gradient_accumulation_steps == 0:
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        # Calculate accuracy (where argmax(out) == targets and targets != 1928)
        value_pred = torch.argmax(value, dim=-1)
        value_acc = (value_pred == true_values).float()
        value_acc = value_acc[true_values != 3].mean()
        acc = (torch.argmax(out, dim=-1) == targets).float()
        acc = acc[targets != 1928].mean()
        
        fens = inp[2]
        if step_counter % 1000 == 0:
            if step_counter == 0:
                tactics_accuracy = 0
                endgame_score = 0
                win_rate_vs_stockfish = 0
                elo = 0
            else:
                #tactics_accuracy = calculate_tactics_accuracy(model, num_positions=1000, is_thinking_model=False, T=0.1)
                #endgame_score = calculate_endgame_score(model, T=0.1, is_thinking_model=False, limit_elo=False, num_positions=100)
                #win_rate_vs_stockfish,elo = main_model_vs_stockfish(model=model,model1_name=f"{step_counter}", temp=0, num_games=40,elo = 1400,use_value=True)
                #win_rate_vs_model = main_model_vs_model(model1=model,model1_name=f"{step_counter}", temp=0.4, num_games=100)
                pass
        else:
            pass
        step_counter += 1
        progress_bar.set_description(f"Loss: {accumulated_loss:.4f} Accuracy: {value_acc.item()}")
        

        if use_wandb:
            log_dict = {
                "loss": accumulated_policy_loss,
                "value_loss": accumulated_value_loss,
                "total_loss": accumulated_loss,
                "accuracy": acc.item(),
                "lr": opt.param_groups[0]["lr"],
                "loss_q": accumulated_q_loss,
            }
            log_dict["value_accuracy"] = value_acc.item()
            log_dict["tactics_accuracy"] = tactics_accuracy
            log_dict["endgame_score"] = endgame_score
            log_dict["win_rate_vs_stockfish"] = win_rate_vs_stockfish
            log_dict["elo"] = elo
            log_dict["epoch"] = epoch
            
            wandb.log(log_dict)
        accumulated_loss = 0.0
        accumulated_value_loss = 0.0
        accumulated_policy_loss = 0.0
        accumulated_q_loss = 0.0
    # Save model checkpoint every 1000 steps
    if (i + 1) % 1000 == 0:
        #pass
        checkpoint_path = f"pretrain/value_only_new_{i+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at step {i+1}")
