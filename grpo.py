import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data.vocab import policy_index
import chess
from torch.utils.tensorboard import SummaryWriter
import time
class StockfishEvaluator:
    def __init__(self, stockfish_path: str = "stockfish/stockfish-ubuntu-x86-64-avx2"):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    def evaluate_move(self, fen: str, move: str) -> float:
        """
        Evaluates a move using Stockfish and returns a reward based on the change in evaluation.
        
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
            return -100.0
        if not chess.Move.from_uci(move) in board.legal_moves:
            return -100.0  # Very low reward for illegal moves
        
        info_before = self.engine.analyse(board, chess.engine.Limit(depth=15))
        eval_before = info_before["score"].relative.score(mate_score=10000)
        
        board.push(chess.Move.from_uci(move))
        info_after = self.engine.analyse(board, chess.engine.Limit(depth=15))
        eval_after = info_after["score"].relative.score(mate_score=10000)
        #print(eval_after,eval_before)
        return (-eval_after - eval_before) / 10  # Normalize score change
    
    def close(self):
        self.engine.quit()


class ChessGRPOTrainer:
    def __init__(self, model, ref_model, optimizer, epsilon=0.2, beta=0, G=64,grad_acc = 1, alpha = 0.0):
        self.model = model  # Current policy (π_θ)
        self.ref_model = ref_model  # Reference model (π_ref)
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.beta = beta
        self.G = G  # Group size
        self.epsilon = 1e-5
        self.grad_acc =  grad_acc
        self.count = 0
        self.alpha = alpha

        self.writer = SummaryWriter()
        self.loss_t = 0
        self.kl_t = 0
        self.rewards_t = 0
        self.format_reward_t=0
        self.move_rewards_t=0
        self.acc_t = 0
        self.legal_rewards_t = 0
        self.stockfish = StockfishEvaluator()
    def generate_sequence(self, board_tensor):
        """Generate a sequence using the current policy."""
        sequences = []
        old_probs = []
        board_expand = board_tensor.unsqueeze(0).expand(self.G,-1,-1,-1,-1).contiguous()
        board_expand = board_expand.view(self.G*board_expand.shape[1],board_expand.shape[2],board_expand.shape[3],board_expand.shape[4])
        sequences,old_probs = self.model.generate_sequence(board_expand)
        return sequences.view(self.G,-1,sequences.shape[1]),old_probs.view(self.G,-1,old_probs.shape[1],old_probs.shape[2]),board_expand

    def compute_rewards(self, sequences, target_moves,fens):
        """Calculate format and move rewards."""
        print(fens)
        target_moves = target_moves[:,0]
        #print(target_moves)
        format_rewards = torch.zeros((sequences.shape[0], sequences.shape[1])) #(G,batch_size)
        move_rewards = torch.zeros((sequences.shape[0], sequences.shape[1])) #(G,batch_size)
        legal_rewards = torch.zeros((sequences.shape[0], sequences.shape[1])) #(G,batch_size)
        accs = 0
        for i in range(sequences.shape[0]):
            for j in range(sequences.shape[1]):
                occurences = {}
                last_occurence = {}
                target_move = target_moves[j].item()
                for k in range(sequences.shape[2]):
                    move = sequences[i,j,k].item()
                    last_occurence[move] = k
                    if move in occurences:
                        occurences[move] += 1
                    else:
                        occurences[move] = 1
                        if move == end_index:
                            break
                move_played= sequences[i,j,k-1].item()
                #print(fens[j],policy_index[move_played])
                legal_reward = self.compute_legal_reward(sequences[i,j,:k+1],fens[j])
                format_reward  = self.compute_format_reward(occurences,last_occurence)
                print(fens[j])
                for move in sequences[i,j,:k+1].cpu().numpy():
                    print(policy_index[move])
                print(format_reward)
                #move_reward,acc = self.compute_move_reward(occurences,last_occurence,target_move)
                move_reward,acc = self.compute_move_reward_bis(occurences,last_occurence,target_move,fens[j],move_played)
                accs += acc
                format_rewards[i,j] = format_reward
                move_rewards[i,j] = move_reward
                legal_rewards[i,j] = legal_reward
        accs = accs / (sequences.shape[0]*sequences.shape[1])
        rewards = format_rewards + 3 *  move_rewards + 0.2 * legal_rewards
        return rewards, format_rewards, move_rewards,legal_rewards,accs

    def compute_move_reward(self,occurences,last_occurence,target_move):
        move_reward = 0
        acc = 0
        if end_think_index in occurences and end_index in occurences and occurences[end_think_index] == 1 and occurences[end_index] == 1 and last_occurence[end_think_index] + 2 == last_occurence[end_index] and target_move in occurences and last_occurence[target_move] + 1 == last_occurence[end_index] : 
            move_reward += 10
            acc += 1

        return move_reward,acc
    
    def compute_move_reward_bis(self,occurences,last_occurence,target_move,fen,move_played):

        acc = 0
        if end_think_index in occurences and end_index in occurences and occurences[end_think_index] == 1 and occurences[end_index] == 1 and last_occurence[end_think_index] + 2 == last_occurence[end_index] and target_move in occurences and last_occurence[target_move] + 1 == last_occurence[end_index] : 
            acc += 1
        move_reward = self.stockfish.evaluate_move(fen,policy_index[move_played])

        #print("move_reward : ", move_reward)
        return move_reward,acc
    def compute_format_reward(self,occurences,last_occurence):
        format_reward = 0
        if end_index not in occurences:
            format_reward -= 5
        #if <thinking> is not played
        if start_think_index not in occurences:

            format_reward -= 5
        # if </thinking> is not played
        if end_think_index not in occurences:

            format_reward -= 5
        if end_variation_index not in occurences:
            format_reward -= 12
        # if <thinking> is played more than once
        if start_think_index in occurences and occurences[start_think_index] > 1:

            format_reward -= 2
        # if </thinking> is played more than once
        if end_think_index in occurences and occurences[end_think_index] > 1:

            format_reward -= 2
        # if <thinking> is played before </thinking>
        if start_think_index in occurences and end_think_index in occurences and occurences[start_think_index] == 1 and  occurences[end_think_index] == 1 and last_occurence[end_think_index] > last_occurence[start_think_index]:

            format_reward += 2
            
        #if </thinking> and end are played exactly once and are separated by just one move
        if end_think_index in occurences and end_index in occurences and occurences[end_think_index] == 1 and occurences[end_index] == 1 and last_occurence[end_think_index] + 2 == last_occurence[end_index]:
            format_reward += 10
        #print("format_reward : ", format_reward)

        if end_variation_index in occurences and occurences[end_variation_index] < 5:
            format_reward -= (5 - occurences[end_variation_index]) * 2


        if end_variation_index in occurences and occurences[end_variation_index] > 5:
            format_reward -= (occurences[end_variation_index] - 5) * 2

        return format_reward


    def compute_legal_reward(self,moves_id,fen):
        moves_id = moves_id.cpu().numpy()
        board = chess.Board()
        reward = 0
        #print(fen)
        for move_id in moves_id:
            #print(policy_index[move_id])
            if move_id == start_think_index or move_id == end_variation_index:
                board.set_fen(fen)
            elif move_id == end_think_index:
                break
            else:
                try:
                    board.push_uci(policy_index[move_id])
                    reward += 1
                except:
                    #print("failed move")
                    reward -= 1
        #print("legal : ", reward)
        return reward
    


    def compute_loss(self, new_logits, ref_logits, advantages, sequences):
        """Compute the loss as the ratio of π_θ(a|s) to π_θ_no_grad(a|s), minus KL(π_θ || π_ref)."""
        new_logits = new_logits[:,:,64:,:]
        ref_logits = ref_logits[:,:,64:,:]

        matches = (sequences == end_index)
        #print(matches.device,sequences.device)
        first_indices = torch.where(matches, torch.arange(sequences.shape[2],device = matches.device).expand_as(sequences), sequences.shape[2])
        first_occurrence = first_indices.min(dim=-1).values

        arange = torch.arange(sequences.shape[2],device = matches.device).view(1, 1, -1)  # Shape (1, 1, 192)
        mask = arange < first_occurrence.unsqueeze(-1)  # Ones before k, zeros after


        advantages = advantages.to("cuda").unsqueeze(-1).unsqueeze(-1)
        #print(advantages)
        new_log_probs = torch.nn.functional.log_softmax(new_logits, dim=-1)

        with torch.no_grad():
            new_log_probs_no_grad = torch.nn.functional.log_softmax(new_logits, dim=-1)

        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

        # Compute probability ratio
        ratios = torch.exp(new_log_probs - new_log_probs_no_grad)  # π_θ(a|s) / (π_θ(a|s) no grad)
        #ratios = torch.softmax(new_logits, dim=-1) / torch.softmax(ref_logits, dim=-1)
        #print(sequences.shape,ratios.shape) #(G,b, 192) (G,b, 192, 1929)
        ratios = ratios.gather(dim=-1, index=sequences.unsqueeze(-1))  # Select the probabilities of the actions taken
        ratios = ratios * mask.unsqueeze(-1)
        # Compute KL divergence: KL(π_θ || π_ref)
        kl_divergence = (torch.exp(new_log_probs) * (new_log_probs - ref_log_probs)).sum(dim=-1)
        # Compute loss
        clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -(torch.min(ratios * advantages, clipped_ratios * advantages)).mean()
        loss = policy_loss / (self.G * 192)  + self.beta * kl_divergence.mean()
        return loss, kl_divergence

    def compute_loss_bis(self, new_logits, ref_logits, advantages, sequences):
        """Compute the loss as the ratio of π_θ(a|s) to π_θ_no_grad(a|s), minus KL(π_θ || π_ref)."""
        new_logits = new_logits[:,:,64:,:]
        ref_logits = ref_logits[:,:,64:,:]

        matches = (sequences == end_index)
        #print(matches.device,sequences.device)
        first_indices = torch.where(matches, torch.arange(sequences.shape[2],device = matches.device).expand_as(sequences), sequences.shape[2])
        first_occurrence = first_indices.min(dim=-1).values

        arange = torch.arange(sequences.shape[2],device = matches.device).view(1, 1, -1)  # Shape (1, 1, 192)
        mask = (arange <= first_occurrence.unsqueeze(-1)).float().unsqueeze(-1)  # Ones before k, zeros after
        advantages = advantages.to("cuda").unsqueeze(-1).unsqueeze(-1)
        '''
        for i in range(2):
            for j in range(4):
                print(sequences[i][j])
        print(advantages)
        #print(advantages)
        '''
        new_log_probs = torch.nn.functional.log_softmax(new_logits, dim=-1)
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        new_log_probs = new_log_probs.gather(dim=-1, index=sequences.unsqueeze(-1))
        ref_log_probs = ref_log_probs.gather(dim=-1, index=sequences.unsqueeze(-1))
        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1

        per_token_loss = torch.exp(new_log_probs - new_log_probs.detach()) * advantages
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * mask).sum(dim=2) / mask.sum(dim=2)).mean()
        mean_kl = ((per_token_kl * mask).sum(dim=2) / mask.sum(dim=2)).mean()

        return loss, mean_kl




    def train_step(self, batch):
        """Process one batch of board positions and target moves."""
        board_tokens_batch, target_moves_batch,fens = batch
        sequences, old_probs,board_tokens_batch = self.generate_sequence(board_tokens_batch)
        rewards, format_reward, move_rewards,legal_rewards,acc  = self.compute_rewards(sequences, target_moves_batch,fens)
        #print(rewards,rewards.shape)
        advantages = (rewards - rewards.mean(dim=0) + self.alpha * (rewards.mean(dim=0)- rewards.mean())) / (rewards.std(dim=0) + 0.1)

        #print(advantages)
        #time.sleep(2)
        #print(rewards.std(dim=0))
        #print(advantages.max(),advantages.min())
        batched = sequences.view(-1,sequences.shape[2])
        new_logits, _ , _ = self.model((board_tokens_batch,batched),compute_loss=True)
        new_logits = new_logits.view(self.G,-1,new_logits.shape[1],new_logits.shape[2])
        with torch.no_grad():
            ref_logits, _ , _ = self.ref_model((board_tokens_batch,batched),compute_loss=True)
        ref_logits = ref_logits.view(self.G,-1,ref_logits.shape[1],ref_logits.shape[2])

        #print(new_logits.shape,ref_logits.shape,advantages.shape,sequences.shape)
        loss,kl = self.compute_loss_bis(new_logits, ref_logits, advantages,sequences)
        loss.backward()
        self.loss_t+= loss.item()
        self.kl_t += kl.mean().item()
        self.rewards_t += rewards.mean().item()
        self.format_reward_t += format_reward.mean().item()
        self.move_rewards_t += move_rewards.mean().item()
        self.legal_rewards_t += legal_rewards.mean().item()
        self.acc_t += acc
        if (self.count+1)%self.grad_acc==0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            wandb.log({"loss":self.loss_t/self.grad_acc, "kl_divergence": self.kl_t/self.grad_acc,\
             "rewards": self.rewards_t/self.grad_acc, "format_rewards":self.format_reward_t/self.grad_acc,\
              "move_rewards": self.move_rewards_t/self.grad_acc,"legal_rewards": self.legal_rewards_t/self.grad_acc, "accuracy": self.acc_t / self.grad_acc})
            self.writer.add_scalar("Loss", self.loss_t/self.grad_acc, self.count)
            self.writer.add_scalar("KL Divergence", self.kl_t/self.grad_acc, self.count)
            self.writer.add_scalar("Rewards", self.rewards_t/self.grad_acc, self.count)
            self.writer.add_scalar("Format Rewards", self.format_reward_t/self.grad_acc, self.count)
            self.writer.add_scalar("Move Rewards", self.move_rewards_t/self.grad_acc, self.count)
            self.writer.add_scalar("Accuracy", self.acc_t / self.grad_acc, self.count)
            
            self.loss_t = 0
            self.kl_t = 0
            self.rewards_t = 0
            self.legal_rewards_t = 0
            self.format_reward_t=0
            self.move_rewards_t=0
            self.acc_t = 0
        self.count+=1
        return loss

if __name__ == "__main__":
    from model import GPT, GPTConfig
    config = GPTConfig()
    config.vocab_size = 1929
    config.block_size = 256


    model = GPT(config).to("cuda")
    reference = GPT(config).to("cuda")
    import wandb
    wandb.init(project="ChessRL-GRPO")
    from data.parse import dir_iterator
    dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
    gen = dir_iterator(dir_path,triple = True)
    #load weights 
    model.load_state_dict(torch.load("fine_tune/checkpoint_step_fine_tune_10000.pt"))
    reference.load_state_dict(torch.load("fine_tune/checkpoint_step_fine_tune_10000.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
    # Initialize trainer
    trainer = ChessGRPOTrainer(model, reference, optimizer)
    
    import copy

    start_think_index = policy_index.index("<thinking>")
    end_think_index = policy_index.index("</thinking>")
    end_variation_index = policy_index.index("end_variation")
    end_index = policy_index.index("end")
    # Train step
    num_steps = 100000
    for _ in range(num_steps):
        batch = next(gen)
        #print(batch)
        loss = trainer.train_step(batch)
        if _ % 100 == 0:
            trainer.ref_model = copy.deepcopy(trainer.model)
        #print(loss.item())
 