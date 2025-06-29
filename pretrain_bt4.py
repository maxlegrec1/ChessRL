import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from model_bis import GPT, GPTConfig
from Leela.test_load_preprocessed import dir_iterator
from tqdm import tqdm
import wandb
import os
import chess
from utils.vocab import policy_index
from metrics.tactics_evaluation_improved import calculate_tactics_accuracy
from metrics.endgame_evaluation import calculate_endgame_score
from metrics.play_move_gemini import main_model_vs_model,main_model_vs_stockfish
config = GPTConfig()
#config.n_layer = 15
#config.n_embd= 1024
#config.n_head = 32
config.vocab_size = 1929
config.block_size = 256

model = GPT(config).to("cuda")
#model.load_state_dict(torch.load("pretrain/model.pt"))
scaler = GradScaler()  # Initialize GradScaler for mixed precision
#dir_path = "data/preprocessed"
dir_path = ["data/preprocessed_higher_quality","data/preprocessed","data/preprocessed_follow"]

opt = torch.optim.Adam(model.parameters(), lr=1e-4)

gen = dir_iterator(dir_path,device = "cuda")

def compute_legal_prob(out,fens,targets,limit_batch = 10):
    #compute legal prob
    #print(out.shape,targets.shape)
    legal_prob = 0
    legal_prob_first_move = 0
    out = out[:limit_batch,63:-1,:]
    targets = targets[:limit_batch,63:-1]
    counter = 0
    softmaxed = torch.nn.functional.softmax(out,dim = -1)
    for i in tqdm(range(out.shape[0])):
        fen = fens[i]
        board = chess.Board(fen)
        for k,target in enumerate(targets[i]):
            if target == 1928:
                break
            for j in range(out.shape[2]):
                move = policy_index[j]
                try:
                    move = chess.Move.from_uci(move)
                    if move in board.legal_moves:
                        legal_prob += softmaxed[i,k,j]
                        if k==0:
                            legal_prob_first_move += softmaxed[i,k,j]
                except:
                    pass
            try:
                board.push_uci(policy_index[target])
                counter+=1      
            except:
                break  
            
    return legal_prob / counter, legal_prob_first_move / limit_batch


use_wandb = True
if use_wandb:
    #wandb.init(project="ChessRL-distill")
    wandb.init(project="ChessRL-pretrain")
    #wandb.watch(model)

gradient_accumulation_steps = 10  # Number of steps to accumulate gradients
num_steps = 300_000*gradient_accumulation_steps
progress_bar = tqdm(range(num_steps))
accumulated_loss = 0.0
step_counter = 0
enable_float16 = True
for i in progress_bar:
    inp,policies,fens = next(gen)
    
    with autocast(device_type='cuda', dtype=torch.float16, enabled= enable_float16):  # Enable mixed precision
        out, _, targets = model(inp, compute_loss=True)
    # Compute loss
    out_truncated = out[:,63:-1,:]
    log_probs = F.log_softmax(out_truncated, dim=-1)
    valid_mask = (policies >= 0).all(dim=-1, keepdim=True)  # shape: (batch, seq_len, 1)


    # Pre-mask invalid policies to zero (safe for KLDiv)
    safe_policies = policies.clone()
    safe_policies[~valid_mask.expand_as(policies)] = 0.0

    # Compute KL divergence
    loss = F.kl_div(log_probs, safe_policies, reduction='none')  # (B, T, V)

    loss = loss * valid_mask  # zero out invalid entries
    # Reduce the loss manually (e.g., mean over valid terms only)
    num_valid = valid_mask.sum().clamp(min=1)  # avoid division by zero
    loss = loss.sum() / num_valid
    # Accuracy computation
    pred = out_truncated.argmax(dim=-1)           # shape: (B, T)
    target = policies.argmax(dim=-1)    # shape: (B, T)
    valid_mask_flat = valid_mask.squeeze(-1)  # shape: (B, T)

    correct = (pred == target) & valid_mask_flat  # boolean tensor
    accuracy = correct.sum().float() / num_valid.float()

    loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
    scaler.scale(loss).backward()
    accumulated_loss += loss.item()
    if (i + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

        #compute metrics
        if step_counter % 400 == 0:
            legal_prob,legal_prob_first_move = compute_legal_prob(out,fens,targets)
            tactics_accuracy = calculate_tactics_accuracy(model, num_positions=1000, is_thinking_model=False, T=0.1)
            endgame_score = calculate_endgame_score(model, T=0.1, is_thinking_model=False, limit_elo=False, num_positions=100)
            win_rate_vs_stockfish,elo = main_model_vs_stockfish(model=model,model1_name=f"{step_counter}", temp=0.1, num_games=40)
            win_rate_vs_model = main_model_vs_model(model1=model,model1_name=f"{step_counter}", temp=0.4, num_games=100)
        else:
            pass
        step_counter+=1
        progress_bar.set_description(f"Loss: {accumulated_loss:.4f} Accuracy: {accuracy.item()}")
        if use_wandb:
            log_dict = {
                "loss": accumulated_loss,
                "accuracy": accuracy.item(),
                "lr": opt.param_groups[0]["lr"],
            }
            log_dict["legal_prob"] = legal_prob
            log_dict["legal_prob_first_move"] = legal_prob_first_move
            log_dict["tactics_accuracy"] = tactics_accuracy
            log_dict["endgame_score"] = endgame_score
            log_dict["win_rate_vs_stockfish"] = win_rate_vs_stockfish
            log_dict["elo"] = elo
            log_dict["win_rate_vs_model"] = win_rate_vs_model    
            wandb.log(log_dict)
        accumulated_loss = 0.0
    
    # Save model checkpoint every 1000 steps
    if (i + 1) % 10000 == 0:
        checkpoint_path = f"pretrain/pretrain_bt4_{i+1}.pt"
        #torch.save(model.state_dict(), checkpoint_path)
        #print(f"Model checkpoint saved at step {i+1}")
