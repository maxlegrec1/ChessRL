import torch
from torch.amp import autocast, GradScaler
from model import GPT, GPTConfig
from fine_tune import gen
from tqdm import tqdm
import wandb
import os
from data.vocab import policy_index
import chess
config = GPTConfig()
config.vocab_size = 1929
config.block_size = 256
model = GPT(config).to("cuda")
model.load_state_dict(torch.load("pretrain/checkpoint_step_40000.pt"))
scaler = GradScaler()  # Initialize GradScaler for mixed precision

  
start_think_index = policy_index.index("<thinking>")
end_think_index = policy_index.index("</thinking>")
end_variation_index = policy_index.index("end_variation")
end_index = policy_index.index("end")


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
            if target == start_think_index:
                continue
            if target == end_think_index:
                continue
            if target == end_variation_index:
                board = chess.Board(fen)
                continue
            if target == end_index:
                continue
            for j in range(out.shape[2]):
                move = policy_index[j]
                try:
                    if move == "end_variation":
                       legal_prob += softmaxed[i,k,j] 
                    else:
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




dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
num_steps = 13000
gen = gen()
gradient_accumulation_steps = 16
use_wandb = True
step_counter = 0
if use_wandb:
    wandb.init(project="ChessRL-fine-tune")
    wandb.watch(model)

progress_bar = tqdm(range(num_steps))
for i in progress_bar:
    inp = next(gen)
    if i == 0:
        print(inp[0].shape)
    with autocast(device_type='cuda', dtype=torch.float16,enabled = False):  # Enable mixed precision
        out, loss, targets = model(inp, compute_loss=True)
    scaler.scale(loss).backward()
    if (i + 1) % gradient_accumulation_steps == 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

        fens = inp[2]

        if step_counter % 100 == 0:
            legal_prob,legal_prob_first_move = compute_legal_prob(out,fens,targets)
        else:
            pass
        step_counter+=1
    
        # Calculate accuracy (where argmax(out) == targets and targets != 1928)
        acc = (torch.argmax(out, dim=-1) == targets).float()
        start_think_acc = acc[targets == start_think_index].mean()
        end_think_acc = acc[targets == end_think_index].mean()
        end_variation_acc = acc[targets == end_variation_index].mean()
        end_acc = acc[targets == end_index].mean()
        acc = acc[targets != 1928].mean()
        progress_bar.set_description(f"Loss: {loss.item()} Accuracy: {acc.item()}")
        if use_wandb:
            wandb.log({"loss": loss.item(), "accuracy": acc.item(), "start_think_acc": start_think_acc.item(), "end_think_acc": end_think_acc.item(), "end_variation_acc": end_variation_acc.item(), "end_acc": end_acc.item(), "legal_prob": legal_prob, "legal_prob_first_move": legal_prob_first_move})
    

checkpoint_path = f"fine_tune/new_{i+1}.pt"
torch.save(model.state_dict(), checkpoint_path)