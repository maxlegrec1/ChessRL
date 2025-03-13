import torch
from model import GPT,GPTConfig
from data.parse import dir_iterator
from grpo_refactored import StockfishEvaluator
from data.vocab import policy_index
from tqdm import tqdm
config = GPTConfig()
config.vocab_size = 1929
config.block_size = 256
device = "cuda"
model1 = GPT(config).to(device)
model1.load_state_dict(torch.load("fine_tune/new_13000.pt"))

model2 = GPT(config).to(device)
model2.load_state_dict(torch.load("GRPO.pt"))
stockfish = StockfishEvaluator()
dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
gen = dir_iterator(dir_path, triple = True)
model1_reward = 0
model2_reward = 0
for _ in tqdm(range(100)):
    inp = next(gen)
    board, moves, fens = inp
    temp = 0.3
    print(f"fen : {fens[0]}")
    id_model1, _ = model1.generate_sequence(board.to(device),T =temp)
    list_moves_model1 = []
    for move in id_model1[0]:
        list_moves_model1.append(policy_index[move])
    id_model2, _ = model2.generate_sequence(board.to(device),T = temp)
    for i,move in enumerate(list_moves_model1):
        if move == "end":
            break
    move_played_model1 = list_moves_model1[i-1]
    list_moves_model2 = []
    for move in id_model2[0]:
        list_moves_model2.append(policy_index[move])
    for i,move in enumerate(list_moves_model2):
        if move == "end":
            break
    move_played_model2 = list_moves_model2[i-1]

    stockfish_eval1 = stockfish.evaluate_move(fens[0],move_played_model1,bypass_ratio=True)

    stockfish_eval2 = stockfish.evaluate_move(fens[0],move_played_model2,bypass_ratio=True)

    print(list_moves_model1)
    print(list_moves_model2)

    print(move_played_model1,stockfish_eval1)
    print(move_played_model2,stockfish_eval2)
    model1_reward += stockfish_eval1
    model2_reward += stockfish_eval2

print(model1_reward)
print(model2_reward)
stockfish.close()