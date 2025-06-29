import os
import chess
import chess.pgn
import numpy as np
import torch
import zstandard as zstd
import io
import matplotlib.pyplot as plt
from tqdm import tqdm
#batch_size = 400 #pretrain
 #grpo
min_length = 20
num_moves = 1
block_size = 256
clip_length = block_size-64

def get_batch(pgn_path, return_fen = False, triple = False,batch_size = 16,all_elo = False):
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
        print(pgn.headers)
        elo = (int(pgn.headers["WhiteElo"])+int(pgn.headers["BlackElo"]))/2
        yield elo

def dir_iterator(dir_path,return_fen = False,triple = False,batch_size = 16, all_elo = False):
    for pgn in os.listdir(dir_path):
        print(pgn)
        pgn_path = os.path.join(dir_path,pgn)
        gen = get_batch(pgn_path,return_fen = return_fen, triple = triple,batch_size = batch_size, all_elo = all_elo)
        while True:
            try:
                yield next(gen)
            except:
                break
   

if __name__ == "__main__":
    gen = dir_iterator("data/compressed_pgns",all_elo = True)
    elos = []
    for _ in tqdm(range(100)):
        elo = next(gen)
        #print(elo)
        elos.append(elo)

    
    # Compute histogram using numpy (optional if using matplotlib directly)
    counts, bins = np.histogram(elos, bins=50)

    # Plot histogram using matplotlib
    plt.figure(figsize=(10, 6))
    plt.hist(elos, bins=50, color='skyblue', edgecolor='black')
    plt.title('ELO Distribution')
    plt.xlabel('ELO')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save the histogram as an image
    #plt.savefig('elo_histogram.png')
    plt.close()