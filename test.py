import torch
from model import GPT,GPTConfig
from data.parse import dir_iterator


config = GPTConfig()
config.vocab_size = 1929
config.block_size = 256
model = GPT(config).to("cuda")
model.load_state_dict(torch.load("checkpoint_step_fine_tune_1101.pt"))

dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
gen = dir_iterator(dir_path)
inp = next(gen)
board, moves = inp
id, logits = model.generate_sequence(board)
print(id.shape)
from data.vocab import policy_index

id = id[0].cpu().numpy()
for move in id:
    print(policy_index[move])
