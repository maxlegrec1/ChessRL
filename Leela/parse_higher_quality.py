import os
import chess
import chess.pgn
import numpy as np
import torch
import zstandard as zstd
import io
from fen_encoder import fen_to_tensor,FenEncoder
from vocab import policy_index
from tqdm import tqdm
import pickle
import subprocess
import time
import signal
import random
#batch_size = 400 #pretrain
 #grpo
min_length = 26
num_moves = 1
block_size = 256
clip_length = block_size-64

def get_batch(pgn_path, return_fen = False, triple = False,batch_size = 16,all_elo = False, elo_threshold = 1600):
    if pgn_path.endswith(".zst"):
        file_compressed = open(pgn_path,'rb')
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(file_compressed)
        f = io.TextIOWrapper(reader, encoding='utf-8')
    else:
        f  = open(pgn_path, "r")
    fen_array = []
    moves_array = []
    policies = []

    while True:
        pgn = chess.pgn.read_game(f)
        if pgn.next()==None:
            continue
        elo = min(int(pgn.headers["WhiteElo"]),int(pgn.headers["BlackElo"]))
        if all_elo:
            elo = 3000
        if elo <= elo_threshold or 'FEN' in pgn.headers.keys() or '960' in pgn.headers['Event'] or 'Odds' in pgn.headers['Event'] or 'house' in pgn.headers['Event']:
            continue
        moves = [move for move in pgn.mainline_moves()]
        if len(moves) < min_length:
            continue
        #check if game ends in mate. if it does, skew start_index closer to end
        dummy_board = chess.Board()
        for move in moves:
            dummy_board.push(move)
        if dummy_board.is_checkmate():
            if random.random() < 0.2:
                start_index = np.random.randint(max(8,len(moves)-15),len(moves)-num_moves)
            else:
                start_index = np.random.randint(8,len(moves)-num_moves)
        #start index is a random int in 8, len(moves)- num_moves
        else:
            start_index = np.random.randint(8,len(moves)-num_moves)
        board = chess.Board()
        for move in moves[:start_index]:
            board.push(move)
        fen = board.fen()

        policies.append(([move.uci() for move in moves[:start_index]],[move.uci() for move in moves[start_index:]]))
        moves = moves[start_index:]
        #print(len(moves))
        fen_array.append(fen)
        moves_array.append(encode_moves(moves))
        if len(fen_array) == batch_size:
            #print(policies)
            try:
                policies = pickle.loads(client_final_final.send_moves(policies))
                print("Policies received from server")
            except Exception as e:
                print("Error in sending moves to server",e)
            policies = clip_and_batch_policies(policies)
            print("done clipping and batching policies")
            if return_fen:
                yield (fen_array)
            elif triple:
                yield (encode_fens(fen_array),clip_and_batch(moves_array),policies, fen_array)
            else:
                yield (encode_fens(fen_array).to("cuda"),clip_and_batch(moves_array).to("cuda"),policies.to("cuda"))

            #for _ in range(len(policies)):
            #    print(policies[_].shape)
            fen_array = []
            moves_array = []
            policies = []
def dir_iterator(dir_path,return_fen = False,triple = False,batch_size = 16, all_elo = False, elo_threshold = 2400):
    for pgn in os.listdir(dir_path):
        print(pgn)
        pgn_path = os.path.join(dir_path,pgn)
        gen = get_batch(pgn_path,return_fen = return_fen, triple = triple,batch_size = batch_size, all_elo = all_elo, elo_threshold= elo_threshold)
        while True:
            try:
                yield next(gen)
            except:
                break
            
def encode_fens(fen_array):
    #encode in pytorch tensor
    #print(fen_array)
    fens = torch.from_numpy(np.array([fen_to_tensor(fen) for fen in fen_array]))
    return fens

def encode_moves(moves_array):
    moves = []
    #print(moves_array)
    for move in moves_array:
        if move.uci() in policy_index:
            move_id = policy_index.index(move.uci())
        else:
            move_id = policy_index.index(move.uci()[:-1])
        moves.append(move_id)
    return torch.from_numpy(np.array(moves))



def encode_moves_bis(moves_array):
    moves = []
    #print(moves_array)
    for move in moves_array:
        if move in policy_index:
            move_id = policy_index.index(move)
        else:
            move_id = policy_index.index(move[:-1])
        moves.append(move_id)
    return torch.from_numpy(np.array(moves))

def clip_and_batch(moves_array,clip = clip_length):
    #clip and batch moves
    moves = torch.full((len(moves_array),clip),1928,dtype = torch.int64)
    for i in range(len(moves_array)):
        if moves_array[i].shape[0] > clip:
            moves[i] = moves_array[i][:clip]
        else:
            moves[i,:moves_array[i].shape[0]] = moves_array[i]
    return moves

def clip_and_batch_policies(policies_array,clip = clip_length):
    #clip and batch moves
    policies = torch.full((len(policies_array),clip,1929),-1000,dtype = torch.float32)
    for i in range(len(policies_array)):
        if policies_array[i].shape[0] > clip:
            policies[i] = torch.from_numpy(policies_array[i][:clip])
        else:
            policies[i,:policies_array[i].shape[0]] = torch.from_numpy(policies_array[i])
    return policies


def load_batch(file_path, device="cuda"):
    """
    Load a compressed and pickled batch file.
    
    Args:
        file_path (str): Path to the .pkl.zst batch file.
        device (str): Device to move tensors to ("cuda" or "cpu").
    
    Returns:
        Tuple of tensors: (fens, moves, policies)
    """
    with open(file_path, 'rb') as f:
        compressed = f.read()
    decompressed = zstd.ZstdDecompressor().decompress(compressed)
    encoded_fens, moves, policies,fens = pickle.loads(decompressed)
    return encoded_fens.to(device), moves.to(device), policies.to(device),fens


if __name__ == "__main__":
    save_path = "data/preprocessed_long"
    max_batches_before_restart = 10_000
    batches_since_restart = 0
    # Adjust the path to the shell script
    server_process = subprocess.Popen(
    ["Leela/launch_server.sh"],
    preexec_fn=os.setsid
    )
    print("server pgid is ",os.getpgid(server_process.pid))
    time.sleep(15)  # Give the server time to start
    import client_final_final
    os.makedirs(save_path, exist_ok=True)
    NUM_BATCH = 10_000_000
    gen = dir_iterator("/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn",batch_size = 128,all_elo = True,triple = True)
    for i in tqdm(range(NUM_BATCH)):


        if batches_since_restart >= max_batches_before_restart:
            print("Restarting server to clear memory leak...")

            # Kill the server process
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            print("Server process terminated, waiting for 60 seconds...")
            time.sleep(30)
            # Restart the server
            server_process = subprocess.Popen(
            ["Leela/launch_server.sh"],
            preexec_fn=os.setsid
            )
            time.sleep(15)

            batches_since_restart = 0
            client_final_final.connect_to_server()

        batches_since_restart += 1
        buffer = io.BytesIO()
        encoded_fens,moves,policies,fens = next(gen)
        #save data
        file_path = os.path.join(save_path, f"batch_{i}.pkl.zst")

        # Serialize the data
        pickle.dump((encoded_fens.cpu(), moves.cpu(), policies.cpu(),fens), buffer, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zstd.ZstdCompressor().compress(buffer.getvalue())

        # Save to file
        with open(file_path, "wb") as f:
            f.write(compressed)

    exit()