
import torch
import chess
import chess.engine
from model_bis import GPT, GPTConfig
from utils.parse import encode_fens
from utils.vocab import policy_index
from tqdm import tqdm
from math import log10
import numpy as np

position = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"


config = GPTConfig()
config.vocab_size = 1929
config.block_size = 256
config.n_layer = 15
config.n_embd= 1024
config.n_head = 32
device = "cuda:0"
model = GPT(config).to(device)
#model.load_state_dict(torch.load("fine_tune/base.pt"))
model.load_state_dict(torch.load("pretrain/pretrain_bt4_hq_10000.pt"))
#model.load_state_dict(torch.load("pretrain/model.pt"))
#model.load_state_dict(torch.load("pretrain/follow_checkpoint_step_160000.pt"))

#move,probs = model.get_move_from_fen(position, T=0.7, device=device, force_legal=True, return_probs=True)
move,probs = model.get_move_from_fen_no_thinking(position, T=0.4, device="cuda", force_legal=True, return_probs=True)
print(move,probs[0].shape)
indexes = np.argsort(-probs[0].cpu().detach().numpy())
for index in indexes[:10]:
    print(policy_index[index],probs[0][index].item())