import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from data.vocab import policy_index
import chess
from torch.utils.tensorboard import SummaryWriter
import time

start_think_index = policy_index.index("<thinking>")
end_think_index = policy_index.index("</thinking>")
end_variation_index = policy_index.index("end_variation")
end_index = policy_index.index("end")

class StockfishEvaluator:
    def __init__(self, stockfish_path: str = "stockfish/stockfish-ubuntu-x86-64-avx2"):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        # Removed occurences tracking as it's no longer used in reward calculation
        # self.occurences = {}
        # self.tot = 0
        # for move in policy_index:
        #     self.occurences[move] = 0

    def evaluate_move(self, fen: str, move: str) -> float:
        """
        Evaluates a move using Stockfish and returns a reward based on the change in evaluation.
        Simplified reward - now primarily based on Stockfish evaluation change.

        Parameters:
            fen (str): The FEN string of the current board position.
            move (str): The move in UCI format (e.g., 'e2e4').

        Returns:
            float: A reward value where higher is better, lower is worse, and very low if illegal.
        """
        board = chess.Board(fen)
        try:
            chess_move = chess.Move.from_uci(move)
        except:
            return -95.0 # Parsing error - very low reward
        if not chess.Move.from_uci(move) in board.legal_moves:
            return -90.0  # Illegal move - very low reward

        # Removed ratio calculation - no longer used
        # self.occurences[move] += 1
        # self.tot += 1
        # ratio = self.occurences[move] / self.tot
        # ratio = min(0.2,ratio) #if ratio is bigger than 20% we cap it
        # ratio = ratio / 0.2 #normalize ratio between 0 and 1

        info_before = self.engine.analyse(board, chess.engine.Limit(depth=15))
        eval_before = info_before["score"].relative.score(mate_score=10000)

        board.push(chess.Move.from_uci(move))
        info_after = self.engine.analyse(board, chess.engine.Limit(depth=15))
        eval_after = info_after["score"].relative.score(mate_score=10000)

        # Simplified reward based on Stockfish evaluation change only
        # Normalized and clipped the score difference.
        score_change = (eval_after - eval_before) / 100.0 # Normalize to a reasonable range
        reward = max(-1.0, min(1.0, score_change)) # Clip reward between -1 and 1

        return reward

    def close(self):
        self.engine.quit()


class ChessGRPOTrainer:
    def __init__(self, model, ref_model, optimizer, config=None):
        self.model = model  # Current policy (π_θ)
        self.ref_model = ref_model  # Reference model (π_ref)
        self.optimizer = optimizer

        # Initialize config with default values - Removed format and legal reward weights
        self.config = {
            "epsilon": 0.2,
            "beta": 0.001,
            "G": 2,
            "grad_acc": 10,
            "alpha": 0.0,
            "epsilon_clip": 1e-5,
            "stockfish_path": "stockfish/stockfish-ubuntu-x86-64-avx2",
            "stockfish_depth": 15,
            "learning_rate": 5e-5,
            "move_reward_weight": 1.0, # Keep move reward weight
            "model_update_frequency": 12,  # Update reference model every 12*grad_acc steps
        }

        # Update config with provided values
        if config:
            self.config.update(config)

        # Set parameters from config
        self.epsilon = self.config["epsilon"]
        self.beta = self.config["beta"]
        self.G = self.config["G"]
        self.epsilon_clip = self.config["epsilon_clip"]
        self.grad_acc = self.config["grad_acc"]
        self.alpha = self.config["alpha"]

        # Initialize counters and metrics - Removed format and legal related metrics
        self.count = 0
        self.loss_t = 0
        self.kl_t = 0
        self.rewards_t = 0
        self.move_rewards_t = 0
        self.legal_probs=0
        self.acc_t = 0 # Keep accuracy metric - clarify its meaning if necessary
        self.response_lengths_sum = 0
        self.response_count = 0

        # Initialize stockfish evaluator
        self.stockfish = StockfishEvaluator(self.config["stockfish_path"])

        # Log config to wandb if enabled
        if 'wandb_log' in globals() and wandb_log:
            wandb.config.update(self.config)

    def generate_sequence(self, board_tensor):
        """Generate a sequence using the current policy."""
        sequences = []
        old_probs = []
        board_expand = board_tensor.unsqueeze(0).expand(self.G,-1,-1,-1,-1).contiguous()
        board_expand = board_expand.view(self.G*board_expand.shape[1],board_expand.shape[2],board_expand.shape[3],board_expand.shape[4])
        sequences,old_probs = self.model.generate_sequence(board_expand)
        return sequences.view(self.G,-1,sequences.shape[1]),old_probs.view(self.G,-1,old_probs.shape[1],old_probs.shape[2]),board_expand

    def compute_rewards(self, sequences, target_moves, fens):
        """Calculate move rewards using Stockfish. Format and Legal rewards are removed."""
        target_moves = target_moves[:,0]
        move_rewards = torch.zeros((sequences.shape[0], sequences.shape[1])) #(G,batch_size)
        response_lengths = []
        accs = 0 # Keep accuracy, but its meaning needs clarification

        for i in range(sequences.shape[0]):
            for j in range(sequences.shape[1]):
                occurences = {}
                last_occurence = {}
                target_move = target_moves[j].item()

                # Find end token and record response length
                response_length = sequences.shape[2]  # Default to max length
                for k in range(sequences.shape[2]):
                    move = sequences[i,j,k].item()
                    last_occurence[move] = k
                    if move in occurences:
                        occurences[move] += 1
                    else:
                        occurences[move] = 1
                    if move == end_index:
                        response_length = k + 1
                        break

                response_lengths.append(response_length)
                move_played = sequences[i,j,k-1].item() # Get the move played (token before end)


                move_reward, acc = self.compute_move_reward_bis(occurences, last_occurence, target_move, fens[j], move_played)
                accs += acc
                move_rewards[i,j] = move_reward


        accs = accs / (sequences.shape[0] * sequences.shape[1])
        # Update response length metrics
        self.response_lengths_sum += sum(response_lengths)
        self.response_count += len(response_lengths)

        # Compute weighted rewards based on config - Now only move_rewards are used
        rewards = self.config["move_reward_weight"] * move_rewards

        return rewards, None, move_rewards, None, accs # format_rewards and legal_rewards are None

    # Removed format-based move reward function: compute_move_reward

    def compute_move_reward_bis(self, occurences, last_occurence, target_move, fen, move_played):
        """Computes move reward using Stockfish. Accuracy is kept but its meaning needs clarification."""
        acc = 0 # Accuracy meaning is unclear and might be removed if not informative
        if end_think_index in occurences and end_index in occurences and occurences[end_think_index] == 1 and occurences[end_index] == 1 and last_occurence[end_think_index] + 2 == last_occurence[end_index] and target_move in occurences and last_occurence[target_move] + 1 == last_occurence[end_index] :
            acc += 1 # Accuracy is still calculated based on format, might need review
        move_reward = self.stockfish.evaluate_move(fen, policy_index[move_played])
        return move_reward, acc

    # Removed format reward function: compute_format_reward
    # Removed legal reward function: compute_legal_reward

    def compute_loss_bis(self, new_logits, ref_logits, advantages, sequences, legal_mask):
        """Compute the loss as the ratio of π_θ(a|s) to π_θ_no_grad(a|s), minus KL(π_θ || π_ref).
           Simplified loss function - legal_loss term removed. Probability ratio calculation reviewed."""
        new_logits = new_logits[:,:,63:-1,:]
        ref_logits = ref_logits[:,:,63:-1,:]

        matches = (sequences == end_index)
        first_indices = torch.where(matches, torch.arange(sequences.shape[2], device=matches.device).expand_as(sequences), sequences.shape[2])
        first_occurrence = first_indices.min(dim=-1).values
        to_consider = first_occurrence != 192

        logits_of_move_played = new_logits.gather(dim=2, index=(first_occurrence-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1929))
        logits_of_move_played = logits_of_move_played.squeeze(2)


        arange = torch.arange(sequences.shape[2], device=matches.device).view(1, 1, -1)  # Shape (1, 1, 192)
        mask = (arange <= first_occurrence.unsqueeze(-1)).float().unsqueeze(-1)  # Ones before k, zeros after
        advantages = advantages.to("cuda").unsqueeze(-1).unsqueeze(-1)

        new_log_probs = torch.nn.functional.log_softmax(new_logits, dim=-1)
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        new_log_probs = new_log_probs.gather(dim=-1, index=sequences.unsqueeze(-1))
        ref_log_probs = ref_log_probs.gather(dim=-1, index=sequences.unsqueeze(-1))

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1

        # PPO Clip loss - using standard clipped ratio with detached probabilities of reference policy
        ratio = torch.exp(new_log_probs - ref_log_probs.detach()) # Probability ratio using detached ref_log_probs (more standard PPO)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
        per_token_loss = -per_token_loss # Negative sign for gradient ascent


        loss = ((per_token_loss * mask).sum(dim=2) / mask.sum(dim=2)).mean()
        mean_kl = ((per_token_kl * mask).sum(dim=2) / mask.sum(dim=2)).mean()


        # Removed legal_loss term - considered redundant and potentially harmful
        # legal_mask_logits = legal_mask + (1-legal_mask) * (-1000)
        # legal_loss = F.kl_div(F.log_softmax(logits_of_move_played,dim=-1),F.softmax(legal_mask_logits,dim=-1),reduction = "none")
        # legal_loss = legal_loss.sum(dim=-1) * to_consider
        # legal_loss = legal_loss.sum() / to_consider.sum()
        # loss = loss + self.config["legal_loss_weight"] * legal_loss

        # Removed legal_loss from print statement
        print(first_occurrence)#, legal_loss)
        legal_prob = F.softmax(logits_of_move_played, dim=-1)
        legal_prob = legal_prob * legal_mask
        legal_prob = legal_prob.sum(dim=-1)
        legal_prob = legal_prob * to_consider
        legal_prob = legal_prob.sum() / to_consider.sum()

        return loss, mean_kl, legal_prob

    def compute_legal_mask(self, fens):
        legal_mask = torch.zeros((len(fens), 1929), dtype=torch.float32, device="cuda")
        for i in range(len(fens)):
            board = chess.Board(fens[i])
            for move in board.legal_moves:
                if move.uci() in policy_index:
                    move_id = policy_index.index(move.uci())
                else:
                    move_id = policy_index.index(move.uci()[:-1])
                legal_mask[i][move_id] = 1
        return legal_mask.unsqueeze(0).expand(self.G, -1, -1)

    def train_step(self, batch):
        """Process one batch of board positions and target moves."""
        board_tokens_batch, target_moves_batch, fens = batch
        legal_mask = self.compute_legal_mask(fens)
        sequences, old_probs, board_tokens_batch = self.generate_sequence(board_tokens_batch)
        rewards, format_reward, move_rewards, legal_rewards, acc = self.compute_rewards(sequences, target_moves_batch, fens)

        max_reward = move_rewards.max(dim=0)[0].mean()
        mean_reward = move_rewards.mean()

        # Simplified advantage calculation - standard normalized advantage
        advantages = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 0.1) # Standard normalized advantage


        batched = sequences.view(-1, sequences.shape[2])
        new_logits, _, _ = self.model((board_tokens_batch, batched), compute_loss=True)
        new_logits = new_logits.view(self.G, -1, new_logits.shape[1], new_logits.shape[2])
        with torch.no_grad():
            ref_logits, _, _ = self.ref_model((board_tokens_batch, batched), compute_loss=True)
        ref_logits = ref_logits.view(self.G, -1, ref_logits.shape[1], ref_logits.shape[2])

        loss, kl, legal_prob = self.compute_loss_bis(new_logits, ref_logits, advantages, sequences, legal_mask)
        loss.backward()

        self.loss_t += loss.item()
        self.kl_t += kl.mean().item()
        self.rewards_t += rewards.mean().item()
        self.move_rewards_t += move_rewards.mean().item() # Only move rewards are relevant now
        self.acc_t += acc
        self.legal_probs += legal_prob.item()

        if (self.count+1) % self.grad_acc == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Calculate average response length
            avg_response_length = self.response_lengths_sum / self.response_count if self.response_count > 0 else 0

            if wandb_log:
                wandb.log({
                    "loss": self.loss_t / self.grad_acc,
                    "kl_divergence": self.kl_t / self.grad_acc,
                    "rewards": self.rewards_t / self.grad_acc,
                    "move_rewards": self.move_rewards_t / self.grad_acc, # Only move rewards are relevant now
                    "accuracy": self.acc_t / self.grad_acc,
                    "legal_prob": self.legal_probs / self.grad_acc,
                    "max_move_reward": max_reward.item(),
                    "mean_move_reward": mean_reward.item(),
                    "avg_response_length": avg_response_length
                })

            # Reset counters - Removed format and legal reward counters
            self.loss_t = 0
            self.kl_t = 0
            self.rewards_t = 0
            self.move_rewards_t = 0
            self.acc_t = 0
            self.legal_probs = 0
            self.response_lengths_sum = 0
            self.response_count = 0

        self.count += 1
        return loss


if __name__ == "__main__":
    from model import GPT, GPTConfig
    import wandb

    # Define configuration - Removed format_reward_weight, legal_reward_weight, legal_loss_weight
    config = {
        "epsilon": 0.2,
        "beta": 0.1,
        "G": 8,
        "grad_acc": 1,
        "alpha": 0.0,
        "epsilon_clip": 1e-5,
        "stockfish_path": "stockfish/stockfish-ubuntu-x86-64-avx2",
        "stockfish_depth": 15,
        "learning_rate": 5e-5,
        "move_reward_weight": 1.0, # Keep move reward weight
        "model_update_frequency": 100,
        "vocab_size": 1929,
        "block_size": 256,
        "dir_path": "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn",
        "num_training_steps": 100000,
    }

    # Initialize model config
    model_config = GPTConfig()
    model_config.vocab_size = config["vocab_size"]
    model_config.block_size = config["block_size"]

    # Initialize models
    model = GPT(model_config).to("cuda")
    reference = GPT(model_config).to("cuda")

    # Initialize wandb
    wandb_log = True
    if wandb_log:
        wandb.init(project="ChessRL-GRPO-SimplifiedReward", config=config, name="SimplifiedRewardSystem") # Project name updated

    # Load dataset
    from data.parse import dir_iterator
    dir_path = config["dir_path"]
    gen = dir_iterator(dir_path, triple=True)

    # Uncomment to load pretrained weights
    model.load_state_dict(torch.load("fine_tune/new_13000.pt"))
    reference.load_state_dict(torch.load("fine_tune/new_13000.pt"))

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Initialize trainer
    trainer = ChessGRPOTrainer(model, reference, optimizer, config)

    import copy

    # Train loop
    num_steps = config["num_training_steps"]
    for step in range(num_steps):
        batch = next(gen)
        loss = trainer.train_step(batch)

        if (step + 1) % (config["model_update_frequency"] * trainer.grad_acc) == 0:
            trainer.ref_model = copy.deepcopy(trainer.model)

        # Removed Stockfish evaluator re-initialization from training loop
        # if (step + 1) % 100 == 0:
        #     trainer.stockfish.close()
        #     trainer.stockfish = StockfishEvaluator(config["stockfish_path"])

    # Close stockfish engine after training
    trainer.stockfish.close()