import torch
from torch.amp import autocast, GradScaler
from new_paradigm.simple import BT4

from utils.parse_value_final import batch_generator
from metrics.tactics_evaluation_improved import calculate_tactics_accuracy
from metrics.endgame_evaluation import calculate_endgame_score
from metrics.play_move_gemini import main_model_vs_model,main_model_vs_stockfish
from tqdm import tqdm
import wandb
import os
import chess
from utils.vocab import policy_index


def train(config):
    """
    Main training loop.
    """


    wandb.init(project=config["wandb_project"], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BT4().to(device)
    #print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])


    use_amp = config["enable_float16"]
    if use_amp:
        scaler = GradScaler()
        print("Using Automatic Mixed Precision (AMP).")
    else:
        print("CUDA not available. Training on CPU without AMP.")


    try:
        #data_generator = leela_data_generator(config_path=config["leela_config_path"])
        data_generator = batch_generator(input_dir="/mnt/2tb/LeelaDataReader/parquet_shuffled_final", batch_size=config["batch_size"], block_size=1, return_fen=False, triple=True, device='cuda')
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print("Starting training...")
    model.train()
    epoch = 0
    accumulated_loss = 0.0
    accumulated_value_loss = 0.0
    accumulated_policy_loss = 0.0
    accumulated_q_loss = 0.0
    step_counter = 0
    for step in tqdm(range(config["num_steps"])):
        try:
            inp = next(data_generator)
        except Exception as e:
            epoch += 1
            del data_generator
            data_generator = batch_generator(config["input_dir"], config["batch_size"], block_size=1, return_fen=False, triple=True, device='cuda')
            inp = next(data_generator)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp,device_type='cuda', dtype=torch.float16):
            out, value, loss_policy, loss_value, loss_q, targets, true_values = model(inp, compute_loss=True)
            loss = loss_value +loss_q+loss_policy
        loss_policy = loss_policy / config["gradient_accumulation_steps"]
        loss_value = loss_value / config["gradient_accumulation_steps"]
        loss_q = loss_q / config["gradient_accumulation_steps"]
        loss = loss / config["gradient_accumulation_steps"]  # Scale loss for accumulation
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()
        accumulated_value_loss += loss_value.item()
        accumulated_policy_loss += loss_policy.item()
        accumulated_q_loss += loss_q.item()
        if (step + 1) % config["gradient_accumulation_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


            # Calculate value accuracy
            value = value.squeeze(1)
            _, predicted_labels = torch.max(value, 1)
            correct_predictions = (predicted_labels == true_values).sum().item()
            batch_size = out.size(0)
            value_accuracy = correct_predictions / batch_size
            
            #Calculate policy accuracy # named accuracy
            acc = (torch.argmax(out, dim=-1) == targets).float()
            acc = acc[targets != 1928].mean()
            
            fens = inp[2]
            if step_counter % 10000 == 0:
                if step_counter == 0:
                    tactics_accuracy = 0
                    endgame_score = 0
                    win_rate_vs_stockfish = 0
                    elo = 0
                    win_rate_vs_stockfish_use_value = 0
                    elo_use_value = 0
                else:
                    tactics_accuracy = calculate_tactics_accuracy(model, num_positions=1000, is_thinking_model=False, T=0.1)
                    endgame_score = calculate_endgame_score(model, T=0.1, is_thinking_model=False, limit_elo=False, num_positions=100)
                    win_rate_vs_stockfish_use_value,elo_use_value = main_model_vs_stockfish(model=model,model1_name=f"{step_counter}", temp=0, num_games=40,elo = max(1400,elo_use_value),use_value=True)
                    win_rate_vs_stockfish,elo = main_model_vs_stockfish(model=model,model1_name=f"{step_counter}", temp=0, num_games=40,elo = max(1400,elo))
                    #win_rate_vs_model = main_model_vs_model(model1=model,model1_name=f"{step_counter}", temp=0.4, num_games=100)
            else:
                pass
            step_counter += 1

            print(f"Step {step}/{config['num_steps']}, Loss: {loss.item():.4f}, Value Accuracy: {value_accuracy:.4f}, Policy Accuracy: {acc.item():.4f}")

            if config["use_wandb"]:
                log_dict = {
                    "loss": accumulated_policy_loss,
                    "value_loss": accumulated_value_loss,
                    "total_loss": accumulated_loss,
                    "accuracy": acc.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss_q": accumulated_q_loss,
                }
                log_dict["value_accuracy"] = value_accuracy
                log_dict["tactics_accuracy"] = tactics_accuracy
                log_dict["endgame_score"] = endgame_score
                log_dict["win_rate_vs_stockfish"] = win_rate_vs_stockfish
                log_dict["elo"] = elo
                log_dict["win_rate_vs_stockfish_use_value"] = win_rate_vs_stockfish_use_value
                log_dict["elo_use_value"] = elo_use_value
                log_dict["epoch"] = epoch
                
                wandb.log(log_dict)
            accumulated_loss = 0.0
            accumulated_value_loss = 0.0
            accumulated_policy_loss = 0.0
            accumulated_q_loss = 0.0
        # Save model checkpoint every 1000 steps
        if (step + 1) % 1000 == 0:
            #pass
            checkpoint_path = f"pretrain/this_time_its_right_{step+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at step {step+1}")


    print("Training finished.")

if __name__ == "__main__":

    config = {
    "learning_rate": 1e-4,
    "num_steps": 50_000_000,
    "wandb_project": "ChessRL-pretrain",
    "batch_size": 1550,
    "use_wandb": True,
    "input_dir": "/mnt/2tb/LeelaDataReader/parquet_shuffled_final",
    "gradient_accumulation_steps": 1,
    "enable_float16": True,
    }
    train(config) 