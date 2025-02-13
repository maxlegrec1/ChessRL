import torch
from model import GPT,GPTConfig
from data.parse import dir_iterator

from data.vocab import policy_index
config = GPTConfig()
config.vocab_size = 1929
config.block_size = 256
model = GPT(config).to("cuda:1")
model.load_state_dict(torch.load("checkpoint_step_fine_tune_1101.pt"))

dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
gen = dir_iterator(dir_path)
inp = next(gen)
board, moves = inp
for i in range(16):
    id, logits = model.generate_sequence(board.to("cuda:1"))
    lists = []
    for move in id[0]:
        lists.append(policy_index[move])
    print(lists)