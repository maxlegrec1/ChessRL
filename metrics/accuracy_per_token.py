import torch
import os
import chess
import chess.pgn
import io
import zstandard as zstd
from model_bis import GPT, GPTConfig
from utils.vocab import policy_index
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
def get_batch(pgn_path,all_elo = False):
    if pgn_path.endswith(".zst"):
        file_compressed = open(pgn_path,'rb')
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(file_compressed)
        f = io.TextIOWrapper(reader, encoding='utf-8')
    else:
        f  = open(pgn_path, "r")
    while True:
        pgn = chess.pgn.read_game(f)
        if pgn.next()==None:
            continue
        elo = min(int(pgn.headers["WhiteElo"]),int(pgn.headers["BlackElo"]))
        if all_elo:
            elo = 3000
        if elo <= 1600 or 'FEN' in pgn.headers.keys() or '960' in pgn.headers['Event'] or 'Odds' in pgn.headers['Event'] or 'house' in pgn.headers['Event'] or 'Bullet' in pgn.headers['Event']:
            continue
        moves = [move for move in pgn.mainline_moves()]
        if len(moves) < 20:
            continue
        dummy_board = chess.Board()
        for move in moves:
            dummy_board.push(move)
        if dummy_board.is_checkmate():
            if random.random() < 0.2:
                start_index = np.random.randint(max(8,len(moves)-15),len(moves)-1)
            else:
                start_index = np.random.randint(8,len(moves)-1)
        #start index is a random int in 8, len(moves)- num_moves
        else:
            start_index = np.random.randint(8,len(moves)-1)
        board = chess.Board()
        for move in moves[:start_index]:
            board.push(move)
        moves = moves[start_index:]
        #board = chess.Board()
        for i, move in enumerate(moves):
            fen = board.fen()
            board.push(move)
            if move.uci() in policy_index:
                move_next = move.uci()
            else:
                move_next = move.uci()[:-1]
            yield (fen,i+start_index,move_next)


def dir_iterator(dir_path,all_elo = False):
    for pgn in os.listdir(dir_path):
        print(pgn)
        pgn_path = os.path.join(dir_path,pgn)
        gen = get_batch(pgn_path,all_elo = all_elo)
        while True:
            try:
                yield next(gen)
            except Exception as e:
                #raise e
                print(e)
                break



if __name__ == "__main__":

    pgn_path = "data/compressed_pgns"
    device = "cuda"

    model_path = "pretrain/model.pt"
    config = GPTConfig()
    config.n_layer = 15
    config.n_embd = 1024
    config.n_head = 32
    config.vocab_size = 1929
    config.block_size = 256

    model = GPT(config).to("cuda")
    model.load_state_dict(torch.load(model_path))

    gen = dir_iterator(pgn_path,all_elo = False)
    buckets_occurence = {i:0 for i in range(0,200)}
    buckets_right = {i:0 for i in range(0,200)}
    for i in tqdm(range(100_000)):
        fen,i,move = next(gen)
        if i >= 200:
            continue
        prediction = model.get_move_from_fen_no_thinking(fen, T = 0)
        #print(fen,i,move, prediction)
        if prediction == move:
            buckets_right[i] += 1
        buckets_occurence[i] += 1


    
    # Compute accuracy per i
    accuracy = {
        i: (buckets_right[i] / buckets_occurence[i]) if buckets_occurence[i] > 0 else 0
        for i in range(200)
    }

    # Plot accuracy per i
    plt.figure(figsize=(10, 5))
    plt.plot(list(accuracy.keys()), list(accuracy.values()), marker='o')
    plt.title("Accuracy per i")
    plt.xlabel("i")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("accuracy_per_i.png")
    plt.close()

    # Plot distribution of i
    plt.figure(figsize=(10, 5))
    plt.bar(list(buckets_occurence.keys()), list(buckets_occurence.values()))
    plt.title("Distribution of i (Occurrence)")
    plt.xlabel("i")
    plt.ylabel("Occurrences")
    plt.grid(True)
    plt.savefig("distribution_per_i.png")
    plt.close()