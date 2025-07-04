import os
import pickle
import random
import torch
import numpy as np
from utils.vocab import policy_index
from utils.parse import clip_and_batch,encode_fens,encode_moves_bis
from tqdm import tqdm


device = "cuda"

files = []


data_path = "data/data_stockfish"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))
data_path = "data/data_antoine/data_stockfish"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))

data_path = "data/data_antoine/data_stockfish2"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))

data_path = "data/new_data"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))

data_path = "data/dce_data"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))

data_path = "new_data/"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))
data_path = "endgame/"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))

data_path = "endgame_parallel/"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))

data_path = "endgame_parallel_weighted/"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))

data_path = "endgame_parallel_zst/"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path,file))

print(len(files))
random.shuffle(files)

def gen():
    for file_path in files:
        print(file_path)
        f = open(file_path,"rb")
        data = pickle.load(f)
        if isinstance(data,dict):
            fens = data["fens"]
            moves = data["var"]
        else: #instanct is list
            fens = [d['fen'] for d in data]
            moves = [d['variations'] for d in data]
        batch_vars = []
        #print(fens)
        for fen,vars in zip(fens,moves):
            variations = []
            first_move_played = vars[0][0]
            shuffled = random.shuffle(vars)
            variations += ["<thinking>"]
            for var in vars:
                variations += var
                variations += ["end_variation"]
            variations += ["</thinking>"]
            variations += [first_move_played]
            variations += ["end"]
            batch_vars.append(encode_moves_bis(variations))

        batch_vars = clip_and_batch(batch_vars)
        batch_fens = encode_fens(fens)
        yield (batch_fens.to("cuda"),batch_vars.to("cuda"),fens)


def compute_rewards(self, sequences, target_moves):
        """Calculate format and move rewards."""
        target_moves = target_moves[:,0]
        format_rewards = torch.zeros((sequences.shape[0], sequences.shape[1])) #(G,batch_size)
        move_rewards = torch.zeros((sequences.shape[0], sequences.shape[1])) #(G,batch_size)
        
        for i in range(sequences.shape[0]):
            for j in range(sequences.shape[1]):
                occurences = {}
                last_occurence = {}
                target_move = target_moves[j].item()
                for k in range(sequences.shape[2]):
                    move = sequences[i,j,k].item()
                    if move in occurences:
                        occurences[move] += 1
                    else:
                        occurences[move] = 1
                    last_occurence[move] = k
                format_reward = 0
                move_reward = 0
                #if end is not played
                if end_index not in occurences:

                    format_reward -= 5
                #if <thinking> is not played
                if start_think_index not in occurences:

                    format_reward -= 5
                # if </thinking> is not played
                if end_think_index not in occurences:

                    format_reward -= 5
                # if <thinking> is played more than once
                if start_think_index in occurences and occurences[start_think_index] > 1:

                    format_reward -= 2
                # if </thinking> is played more than once
                if end_think_index in occurences and occurences[end_think_index] > 1:

                    format_reward -= 2
                # if end is played more than once
                if end_index in occurences and occurences[end_index] > 1:

                    format_reward -= 2
                # if <thinking> is played before </thinking>
                if start_think_index in occurences and end_think_index in occurences and occurences[start_think_index] == 1 and  occurences[end_think_index] == 1 and last_occurence[end_think_index] > last_occurence[start_think_index]:

                    format_reward += 2
                #if the correct move is played
                if target_move in occurences:

                    move_reward += 5
                #if the target move is played after the last think
                if end_think_index in occurences and occurences[end_think_index] == 1 and target_move in last_occurence and last_occurence[target_move] > last_occurence[end_think_index]: 

                    move_reward += 10
                #if the target move is played before the end
                if end_index in occurences and occurences[end_index] == 1 and target_move in last_occurence and last_occurence[target_move] < last_occurence[end_index]: 

                    move_reward += 10
                
                #if </thinking> and end are played exactly once and are separated by just one move
                if end_think_index in occurences and end_index in occurences and occurences[end_think_index] == 1 and occurences[end_index] == 1 and last_occurence[end_think_index] + 2 == last_occurence[end_index]:
                    format_reward += 10
                format_rewards[i,j] = format_reward
                move_rewards[i,j] = move_reward
        rewards = format_rewards + 14*move_rewards
        return rewards, format_rewards, move_rewards


def gen2(multiple = 8):
    sub_gen = gen()
    while True:
        batch_fens = []
        batch_vars = []
        fens = []
        for _ in range(multiple):
            try:
                b,v,f = next(sub_gen)
            except StopIteration:
                return 
            batch_fens.append(b)
            batch_vars.append(v)
            fens+=f
        batch_fens = torch.cat(batch_fens,dim = 0)
        batch_vars = torch.cat(batch_vars,dim = 0)
        yield(batch_fens,batch_vars,fens)

if __name__ == "__main__":

    g = gen()

    boards,moves,fens = next(g)
    #print(boards.shape)
    #print( moves[0])
    moves_human = [policy_index[move] for move in  moves[0]]
    print(fens[0], moves_human)