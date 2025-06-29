import torch
from torch.amp import autocast, GradScaler
from model_bis import GPT, GPTConfig
from utils.parse import dir_iterator
from tqdm import tqdm
import wandb
import os
import chess
from utils.vocab import policy_index
config = GPTConfig()
config.n_layer = 15
config.n_embd= 1536
config.n_head = 32
config.vocab_size = 1929
config.block_size = 256

model = GPT(config).to("cuda")
model.load_state_dict(torch.load("pretrain/follow_extra_long_checkpoint_step_60000.pt"))
scaler = GradScaler()  # Initialize GradScaler for mixed precision
dir_path = "data/compressed_pgns"
#dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
#dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/casual_pgn"
opt = torch.optim.Adam(model.parameters(), lr=5e-5)

gen = dir_iterator(dir_path,triple = True,batch_size = 200, all_elo = False)

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
    wandb.init(project="ChessRL-pretrain",id="65ni8u3c", resume="allow")
    #wandb.watch(model)

gradient_accumulation_steps = 2  # Number of steps to accumulate gradients
num_steps = 10_000_000*gradient_accumulation_steps
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
        #compute legal prob
        if step_counter % 100 == 0:
            legal_prob,legal_prob_first_move = compute_legal_prob(out,fens,targets)
        else:
            pass
        step_counter+=1
        progress_bar.set_description(f"Loss: {accumulated_loss:.4f} Accuracy: {acc.item()}")
        if use_wandb:
            wandb.log({"loss": accumulated_loss, "accuracy": acc.item(), "lr": opt.param_groups[0]["lr"], "legal_prob": legal_prob, "legal_prob_first_move": legal_prob_first_move})
        accumulated_loss = 0.0
    
    # Save model checkpoint every 1000 steps
    if (i + 1) % 10000 == 0:
        checkpoint_path = f"pretrain/large_checkpoint_step_{i+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at step {i+1}")
