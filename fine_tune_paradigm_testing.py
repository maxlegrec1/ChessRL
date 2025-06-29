import torch
from torch.amp import autocast, GradScaler

from gen_paradigm import gen as generator
from tqdm import tqdm
import wandb
import os
from utils.vocab import policy_index
import chess
from new_paradigm.start_v2 import ReasoningTransformer

if __name__ == "__main__":

    model = ReasoningTransformer()
    model.load_state_dict(torch.load("fine_tune/fine_tune_paradigm_3001.pt"))
    model.to("cuda")

    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    
    start_think_index = policy_index.index("<thinking>")
    end_think_index = policy_index.index("</thinking>")
    end_variation_index = policy_index.index("end_variation")
    end_index = policy_index.index("end")


    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    gen = generator(batch_size = 1)
    step_counter = 0
    use_float16 = True

    for i in range(1):

        inp = next(gen)


        out, loss, targets = model(inp, compute_loss=True)
        fens = inp[2]
        # Calculate accuracy (where argmax(out) == targets and targets != 1928)
        acc = (torch.argmax(out, dim=-1) == targets).float()
        start_think_acc = acc[targets == start_think_index].mean()
        end_think_acc = acc[targets == end_think_index].mean()
        end_variation_acc = acc[targets == end_variation_index].mean()
        end_acc = acc[targets == end_index].mean()
        acc = acc[targets != 1928].mean()

        #calculate accuracy of move played
        out_ = out[:,:-1,:]
        targets_ = targets[:,:-1]
        indexes_of_end = (targets_ == end_index).float().argmax(dim=1) - 1 
        out_ = out_[torch.arange(out_.shape[0]),indexes_of_end,:]
        targets_ = targets_[torch.arange(targets_.shape[0]),indexes_of_end]
        acc_move_played = (torch.argmax(out_, dim=-1) == targets_).float().mean()



