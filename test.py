import torch
from model_bis import GPT,GPTConfig
from data.parse import dir_iterator
from data.vocab import policy_index
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
def calculate_metrics(models,gen,Trainer,device = "cuda",num_steps = 100, depth = 15, name = 0,temp = 1, raw = False):
    models_rewards = [0]* len(models)
    plots = [[] for _ in range(len(models))]
    for _ in tqdm(range(num_steps)):
        inp = next(gen)
        board, moves, fens = inp
        #print(f"fen : {fens[0]}")
        for j,model in enumerate(models):
            if raw:
                id_model, _ = model.generate_sequence_raw(board.to(device),T =temp)
            else:
                id_model, _ = model.generate_sequence(board.to(device),T =temp)
            list_moves_model = []
            for i,move in enumerate(id_model[0]):
                list_moves_model.append(policy_index[move])
                print(policy_index[move])
                if policy_index[move]== "end":
                    break
            move_played_model = list_moves_model[i-1]
            stockfish_eval = Trainer.stockfish.evaluate_move(fens[0],move_played_model, depth = depth)
            print(stockfish_eval)
            if stockfish_eval > 100:
                print(stockfish_eval,Trainer.stockfish.before,Trainer.stockfish.after)
            models_rewards[j] += stockfish_eval
            plots[j].append(models_rewards[j])
        Trainer.stockfish.reset_hash_table()
        #print(len(plots[0]))
    print(models_rewards)
    for i in range(len(models)):
        models_rewards[i] = models_rewards[i]/num_steps
        plots[i] = np.array(plots[i])
        print(plots[i].shape)
        plt.plot(plots[i],label=f'Series_{i}')

    plt.legend()
    plt.title("Reward")    
    plt.savefig(f"test_{name}.png")
    plt.clf()
    return models_rewards


if __name__ == "__main__":
    model_config = GPTConfig()
    model_config.vocab_size = 1929
    model_config.block_size = 256
    device = "cuda"
    # Initialize models
    num_models = 1
    models = [GPT(model_config).to(device) for _ in range(num_models)]
    models[0].load_state_dict(torch.load("fine_tune/news_2400.pt"))
    #models[1].load_state_dict(torch.load("999_GRPO.pt"))
    #models[2].load_state_dict(torch.load("1999_GRPO.pt"))
    #models[3].load_state_dict(torch.load("2999_GRPO.pt"))
    gen = dir_iterator("/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn", triple=True,batch_size = 1)

    from grpo_refactored import ChessGRPOTrainer
    Trainer = ChessGRPOTrainer(None,None,None)

    models_rewards = calculate_metrics(models,gen,Trainer,device,num_steps = 1,name = 0,temp = 0.01, raw = True)
    #models_rewards = calculate_metrics(models,gen,Trainer,device,num_steps = 10,name = 1)
    Trainer.stockfish.close()
    print(models_rewards)