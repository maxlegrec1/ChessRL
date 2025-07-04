import torch
from torch.amp import autocast, GradScaler
from model_bis import GPT, GPTConfig
from utils.parse import dir_iterator
from utils.parse_improved import batch_generator
from tqdm import tqdm
import wandb
import os
import chess
from utils.vocab import policy_index
from metrics.tactics_evaluation_improved import calculate_tactics_accuracy
from metrics.endgame_evaluation import calculate_endgame_score
from metrics.play_move_gemini import main_model_vs_model,main_model_vs_stockfish
config = GPTConfig()

config.vocab_size = 1929
config.block_size = 256

model = GPT(config).to("cuda")
#model.load_state_dict(torch.load("pretrain/40M_Leela_T80_Data_300000.pt"))
scaler = GradScaler()  # Initialize GradScaler for mixed precision
#dir_path = "data/compressed_pgns"
dir_path = "/media/maxime/385e67e0-8703-4723-a0bc-af3a292fd030/stockfish_data/Leela_training"
#dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
#dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/casual_pgn"
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

#gen = dir_iterator(dir_path,triple = True,batch_size = 400, all_elo = False)
gen = batch_generator(dir_path, batch_size=400, return_fen=False, triple=True, device='cuda')
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

def log_accuracy_by_depth(out, targets):
    max_depth = 190
    depth_interval = 10
    accuracies = {}
    preds = torch.argmax(out, dim=-1)
    for d in range(63, max_depth + 1 - 63, depth_interval):
        if d >= targets.shape[1]:
            break
        mask = (targets[:, d] != 1928)
        if mask.sum() == 0:
            continue
        acc_d = (preds[:, d] == targets[:, d]).float()[mask].mean()
        accuracies[f"accuracy_depth_{d-63}"] = acc_d.item()
    return accuracies

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
        out, loss, targets = model(inp, compute_loss=True)
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
        # Compute metrics
        if step_counter % 10000 == 0:
            legal_prob, legal_prob_first_move = compute_legal_prob(out, fens, targets)
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
            log_dict["legal_prob"] = legal_prob
            log_dict["legal_prob_first_move"] = legal_prob_first_move
            log_dict["tactics_accuracy"] = tactics_accuracy
            log_dict["endgame_score"] = endgame_score
            log_dict["win_rate_vs_stockfish"] = win_rate_vs_stockfish
            log_dict["elo"] = elo
            log_dict["win_rate_vs_model"] = win_rate_vs_model    
            log_dict.update(log_accuracy_by_depth(out, targets))
            wandb.log(log_dict)
        accumulated_loss = 0.0
    
    # Save model checkpoint every 1000 steps
    if (i + 1) % 100000 == 0:
        pass
        #checkpoint_path = f"pretrain/40M_Leela_T80_Data_{300000+i+1}.pt"
        #torch.save(model.state_dict(), checkpoint_path)
        #print(f"Model checkpoint saved at step {i+1}")
