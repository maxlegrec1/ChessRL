
import chess
import torch
from model_bis import GPT, GPTConfig
from tqdm import tqdm

tactics_path = "data/tactic_evals.csv"

def calculate_tactics_accuracy(model, file = tactics_path,num_positions = 1000,is_thinking_model=True, T =1):
    """
    Calculate the accuracy of the model on a given file.
    """
    #create tqdm bar of size num_positions
    bar = tqdm(total=num_positions)
    bar.set_description("Calculating accuracy")
    with open(file, "r") as f:
        lines = f.readlines()
    #skip first line
    lines = lines[1:]
    correct = 0
    total = 0
    for line in lines:
        if total >= num_positions:
            break
        parts = line.strip().split(",")
        fen = parts[0]
        correct_move = parts[2]
        if correct_move == "": #null move
            continue
        bar.update(1)
        if is_thinking_model:
            move = model.get_move_from_fen(fen, T = T)
        else:
            move = model.get_move_from_fen_no_thinking(fen, T = T)
        
        if move == correct_move:
            correct += 1
            print(fen,correct_move,"found")
        else:
            print(fen,correct_move,"not found")    
        total += 1
    
    return correct / total

if __name__ == "__main__":
    config = GPTConfig()

    #config.n_layer = 15
    #config.n_embd= 1024
    #config.n_head = 32
    config.vocab_size = 1929
    config.block_size = 256


    model = GPT(config).to("cuda")
    #model.load_state_dict(torch.load("pretrain/follow_checkpoint_step_160000.pt"))
    #model.load_state_dict(torch.load("pretrain/small_pretrain_bt4_hq_10000.pt"))
    #model.load_state_dict(torch.load("pretrain/pretrain_bt4_hq_10000.pt"))
    #model.load_state_dict(torch.load("pretrain/pretrain_bt4_40000.pt"))
    #13.6,16.4,15.4,14.4,15.5,17.7,17,19.3,21.1,19.9

    num_games = 1000
    accuracy = calculate_tactics_accuracy(model, tactics_path, num_games, is_thinking_model=False,T = 0.1)

    print(f"Accuracy: {accuracy * 100:.2f}%")