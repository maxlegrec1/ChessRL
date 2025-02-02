
dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
from data.parse import dir_iterator
from stockfish import Stockfish
import numpy as np
import random
stockfish_binary = "stockfish/stockfish-ubuntu-x86-64-avx2"
stockfish = Stockfish(path = stockfish_binary, depth = 15)

stockfish.update_engine_parameters({"MultiPV":5, "Threads" : 2 })

gen = dir_iterator(dir_path,return_fen = True)

print(stockfish.is_fen_valid("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"))

def generate_variations(fen, num_var = 5,min_len_var = 3, max_len_var = 8,top_k = 20):
    """
    For each FEN in the generator, set the position in Stockfish and get the top variations.
    Returns a dictionary mapping each FEN to its list of variations.
    Each variation is a dictionary with keys like 'Move', 'Centipawn', and 'Mate' (if applicable).
    """
    fen_variations = {}
    
    stockfish.set_fen_position(fen)
    best_first_moves = stockfish.get_top_moves(top_k)
    best_move = best_first_moves.pop(0)["Move"]
    num_var_max = min(num_var-1,len(best_first_moves))
    other_vars = random.sample(best_first_moves,num_var_max)
    other_vars = [move["Move"] for move in other_vars]
    first_moves = [best_move] + other_vars
    results = [[] for _ in range(len(first_moves))]
    var_lengths = np.random.randint(min_len_var-1, max_len_var,size=(num_var))
    for k,(first_move,var_length) in enumerate(zip(first_moves,var_lengths)):
        results[k].append(first_move)
        stockfish.make_moves_from_current_position([first_move])
        results[k].extend(make_stockfish_play(var_length, fen))
    return results

def make_stockfish_play(k, fen):
    moves = []
    for _ in range(k):
        move = stockfish.get_best_move()
        if move is None:
            move = "end"
            moves.append(move)
            break
        moves.append(move)
        stockfish.make_moves_from_current_position([move])
    stockfish.set_fen_position(fen)
    return moves
    
import time
from tqdm import tqdm
import pickle
for i in tqdm(range(10000)):
    fens = next(gen) 
    batch = {"fens" : fens, "var" : []}
    for fen in tqdm(fens):
        fen_variations = generate_variations(fen)
        batch["var"].append(fen_variations)
    
    pickle.dump(batch, open(f"data_stockfish/{i}.pkl","wb"))