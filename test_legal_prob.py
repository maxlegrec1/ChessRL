import torch
from model_bis import GPT,GPTConfig
from utils.parse import dir_iterator
from utils.vocab import policy_index
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import chess
def calculate_metrics(model,gen,Trainer,device = "cuda",num_steps = 100, depth = 15,temp = 1, raw = False):
    global_legal_prob = 0
    for _ in tqdm(range(num_steps)):
        inp = next(gen)
        board, moves, fens = inp
        #print(f"fen : {fens[0]}")
        if raw:
            id_model, probs = model.generate_sequence_raw(board.to(device),T =temp)
        else:
            id_model, probs = model.generate_sequence(board.to(device),T =temp)
        list_moves_model = []
        for i,move in enumerate(id_model[0]):
            list_moves_model.append(policy_index[move])
            print(policy_index[move])
            if policy_index[move]== "end":
                break
        move_played_model = list_moves_model[i-1]
        probs = probs[0][i-2]# since we are on the probabilities.
        print(probs)
        board = chess.Board(fens[0])
        legal_prob = 0
        for legal_move in board.legal_moves:
            if legal_move.uci()[-1] == 'n':
                legal_move_uci = legal_move.uci()[:-1]
            else:
                legal_move_uci = legal_move.uci()
            legal_prob += probs[policy_index.index(legal_move_uci)]
        #print(len(plots[0]))
        global_legal_prob += legal_prob
    return global_legal_prob/ num_steps


if __name__ == "__main__":
    model_config = GPTConfig()
    model_config.vocab_size = 1929
    model_config.block_size = 256
    device = "cuda"
    # Initialize models
    model = GPT(model_config).to(device)
    model.load_state_dict(torch.load("grpo_experiment/new_4500.pt"))

    gen = dir_iterator("/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn", triple=True,batch_size = 1)

    from grpo_refactored import ChessGRPOTrainer
    Trainer = ChessGRPOTrainer(None,None,None)

    legal_prob = calculate_metrics(model,gen,Trainer,device,num_steps = 1000,temp = 1, raw = True)
    #models_rewards = calculate_metrics(models,gen,Trainer,device,num_steps = 10,name = 1)
    Trainer.stockfish.close()
    print(legal_prob)