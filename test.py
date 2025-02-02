import torch
from model import GPT,GPTConfig
from data.parse import dir_iterator


config = GPTConfig()
config.vocab_size = 1929
config.block_size = 256
model = GPT(config).to("cuda")

dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
gen = dir_iterator(dir_path)
inp = next(gen)
print(inp[0].shape,inp[1].shape)
out = model(inp)[0]

print(out.shape)