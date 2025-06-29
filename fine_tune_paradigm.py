import torch
from torch.amp import autocast, GradScaler

from load_fine_tune_parquet import gen as generator
from tqdm import tqdm
import wandb
import os
from utils.vocab import policy_index
import chess
from metrics.tactics_evaluation_improved import calculate_tactics_accuracy
from metrics.endgame_evaluation import calculate_endgame_score
from new_paradigm.start_v4 import ReasoningTransformer

if __name__ == "__main__":

    model = ReasoningTransformer()
    model.to("cuda")

    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    
    start_think_index = policy_index.index("<thinking>")
    end_think_index = policy_index.index("</thinking>")
    end_variation_index = policy_index.index("end_variation")
    end_index = policy_index.index("end")


    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_steps = 150000
    gen = generator(batch_size = 20)
    gradient_accumulation_steps = 1
    use_wandb = True
    step_counter = 0
    epoch = 0
    use_float16 = True
    if use_wandb:
        wandb.init(project="ChessRL-fine-tune")
        wandb.watch(model)

    progress_bar = tqdm(range(num_steps * gradient_accumulation_steps ))
    for i in progress_bar:
        try:
            inp = next(gen)
            if i ==0:
                print(inp[0].shape)
        except:
            #reload gen
            epoch += 1
            gen = generator(batch_size = 160)
            inp = next(gen)
        with autocast(device_type='cuda', dtype=torch.float16,enabled = use_float16):  # Enable mixed precision
            out, loss, targets = model(inp, compute_loss=True)
        scaler.scale(loss).backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            fens = inp[2]

            if step_counter % 10000 == 0:
                endgame_score = calculate_endgame_score(model,T=0.1,is_thinking_model=True,limit_elo=False,num_positions=100)
                tactics_accuracy = calculate_tactics_accuracy(model,num_positions=100,is_thinking_model=True,T=0.1,device="cuda")
            else:
                pass
            step_counter+=1
        
            # Calculate accuracy (where argmax(out) == targets and targets != 1928)
            acc = (torch.argmax(out, dim=-1) == targets).float()
            start_think_acc = acc[targets == start_think_index].mean()
            end_think_acc = acc[targets == end_think_index].mean()
            end_variation_acc = acc[targets == end_variation_index].mean()
            end_acc = acc[targets == end_index].mean()
            accuracy = acc[targets != 1928].mean()

            #calculate accuracy of first move of variations (where targets is end_variation_index or targets is start_think_index)
            first_moves_indexes = (targets == end_variation_index ) | (targets == start_think_index)
            #shift that to the right by one
            first_moves_index = torch.roll(first_moves_indexes, shifts=1, dims=1) & (targets != end_think_index) & (targets != 1928)

            first_moves_acc = acc[first_moves_index].mean()


            #calculate accuracy of move played
            out_ = out[:,:-1,:]
            targets_ = targets[:,:-1]
            indexes_of_end = (targets_ == end_index).float().argmax(dim=1) - 1 
            out_ = out_[torch.arange(out_.shape[0]),indexes_of_end,:]
            targets_ = targets_[torch.arange(targets_.shape[0]),indexes_of_end]
            acc_move_played = (torch.argmax(out_, dim=-1) == targets_).float().mean()


            progress_bar.set_description(f"Loss: {loss.item()} Accuracy: {accuracy.item()}")
            if use_wandb:
                log_dict = {}
                log_dict["loss"] = loss.item()
                log_dict["accuracy"] = accuracy.item()
                log_dict["start_think_acc"] = start_think_acc.item()
                log_dict["end_think_acc"] = end_think_acc.item()
                log_dict["end_variation_acc"] = end_variation_acc.item()
                log_dict["end_acc"] = end_acc.item()
                log_dict["acc_move_played"] = acc_move_played.item()
                log_dict["first_moves_acc"] = first_moves_acc.item()
                log_dict["epoch"] = epoch
                log_dict["tactics_accuracy"] = tactics_accuracy
                log_dict["endgame_score"] = endgame_score
                wandb.log(log_dict)
        if i % 10000 == 0:
            checkpoint_path = f"fine_tune/V4_{i+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")    

    checkpoint_path = f"fine_tune/V4_{i+1}.pt"
    torch.save(model.state_dict(), checkpoint_path)



