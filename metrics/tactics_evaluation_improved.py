
import chess
import torch
from model_bis import GPT, GPTConfig
from tqdm import tqdm
from utils.vocab import policy_index
tactics_path = "data/lichess_db_puzzle.csv"

def calculate_tactics_accuracy(model, file=tactics_path, num_positions=1000, is_thinking_model=True, T=1,device = "cuda"):
    """
    Calculate the accuracy of the model on a given file.
    Displays a tqdm progress bar with running accuracy.
    """
    with open(file, "r") as f:
        lines = f.readlines()[1:]  # Skip header

    correct = 0
    total = 0

    bar = tqdm(total=num_positions, desc="Calculating accuracy", dynamic_ncols=True)

    for line in lines:
        if total >= num_positions:
            break

        parts = line.strip().split(",")
        fen = parts[1]
        correct_moves = parts[2].split(" ")

        assert len(correct_moves) % 2 == 0, "Expected even number of moves (engine/player pairs)"
        total += 1
        board = chess.Board(fen)

        success = True
        for i in range(len(correct_moves) // 2):
            engine_move = correct_moves[i * 2]
            board.push_uci(engine_move)
            fen = board.fen()
            target_move = correct_moves[i * 2 + 1]

            # Strip promotion if needed
            target_parsed = target_move if target_move in policy_index else target_move[:-1]

            if is_thinking_model:
                move_played = model.get_move_from_fen(fen, T=T,device = device)
            else:
                move_played = model.get_move_from_fen_no_thinking(fen, T=T,device = device)

            if move_played != target_parsed:
                success = False
                break
            board.push_uci(target_move)

        if success:
            correct += 1

        acc = correct / total
        bar.set_postfix_str(f"Accuracy: {acc:.2%}")
        bar.update(1)

    bar.close()
    return correct / total if total > 0 else 0


if __name__ == "__main__":
    config = GPTConfig()

    config.n_layer = 15
    config.n_embd= 1024
    config.n_head = 32
    config.vocab_size = 1929
    config.block_size = 256

    device = "cuda"  # Use the appropriate device
    model = GPT(config).to(device)
    #model.load_state_dict(torch.load("pretrain/follow_checkpoint_step_160000.pt"))
    #model.load_state_dict(torch.load("pretrain/small_pretrain_bt4_hq_10000.pt"))
    #model.load_state_dict(torch.load("pretrain/pretrain_bt4_hq_10000.pt"))
    #model.load_state_dict(torch.load("pretrain/pretrain_bt4_40000.pt"))
    model.load_state_dict(torch.load("pretrain/model.pt"))
    #13.6,16.4,15.4,14.4,15.5,17.7,17,19.3,21.1,19.9

    num_games = 1000
    accuracy = calculate_tactics_accuracy(model, tactics_path, num_games, is_thinking_model=False,T = 0.1,device = device)

    print(f"Accuracy: {accuracy * 100:.2f}%")