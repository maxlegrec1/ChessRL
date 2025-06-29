import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from utils.vocab import policy_index
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
        self.occurences = {}
        self.tot = 0
        for move in policy_index:
            self.occurences[move] = 0
    
    def evaluate_move(self, fen: str, move: str, depth = 15) -> float:
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
            return -95.0
        if not chess.Move.from_uci(move) in board.legal_moves:
            return -90.0  # Very low reward for illegal moves
        self.occurences[move] += 1
        self.tot += 1
        ratio = self.occurences[move] / self.tot
        print(ratio)
        ratio = min(0.2,ratio) #if ratio is bigger than 20% we cap it
        ratio = ratio / 0.2 #normalize ratio between 0 and 1
        info_before = self.engine.analyse(board, chess.engine.Limit(depth=15))
        eval_before = info_before["score"].relative.score(mate_score=10000)
        
        board.push(chess.Move.from_uci(move))
        info_after = self.engine.analyse(board, chess.engine.Limit(depth=15))
        eval_after = info_after["score"].relative.score(mate_score=10000)
        #print(eval_after,eval_before)
        if bypass_ratio:
            return max(-80,(-eval_after - eval_before) / 10)
        return -80 * (ratio) + (1- ratio)* max(-80,(-eval_after - eval_before) / 10)  # Normalize score change
    
    def close(self):
        self.engine.quit()


class ChessGRPOTrainer:
    def __init__(self, model, ref_model, optimizer, epsilon=0.2, beta=0.001, G=2,grad_acc = 10, alpha = 0.0):
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
        #self.writer = SummaryWriter()
        self.loss_t = 0
        self.kl_t = 0
        self.rewards_t = 0
        self.format_reward_t=0
        self.move_rewards_t=0
        self.acc_t = 0
        self.legal_rewards_t = 0
        self.legal_probs = 0
        self.stockfish = StockfishEvaluator()
    def generate_sequence(self, board_tensor):
        """Generate a sequence using the current policy."""
        sequences = []
        old_probs = []
        board_expand = board_tensor.unsqueeze(0).expand(self.G,-1,-1,-1,-1).contiguous()
        board_expand = board_expand.view(self.G*board_expand.shape[1],board_expand.shape[2],board_expand.shape[3],board_expand.shape[4])
        print(board_expand.shape)
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
                if format_reward!=12:
                    legal_reward = -160
                print(fens[j])
                print(policy_index[move_played])
                
                
                #move_reward,acc = self.compute_move_reward(occurences,last_occurence,target_move)
                move_reward,acc = self.compute_move_reward_bis(occurences,last_occurence,target_move,fens[j],move_played)
                if end_variation_index in sequences[i,j,:k+1].cpu().numpy():
                    for move in sequences[i,j,:k+1].cpu().numpy():
                        print(policy_index[move])
                accs += acc
                if format_reward <0:
                    move_reward = -100
                print(format_reward)
                print(move_reward)
                format_rewards[i,j] = format_reward
                move_rewards[i,j] = move_reward
                legal_rewards[i,j] = legal_reward

        accs = accs / (sequences.shape[0]*sequences.shape[1])
        #rewards = format_rewards #+ 3 *  move_rewards + 0.2 * legal_rewards
        rewards = format_rewards + move_rewards# + legal_rewards
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
        board.set_fen(fen)
        reward = 0
        legal_moves_played = set()
        #print(fen)
        for move_id in moves_id:
            #print(policy_index[move_id])
            if move_id == end_think_index:
                break
            else:
                try:
                    move = policy_index[move_id]
                    if chess.Move.from_uci(move) in board.legal_moves:
                        reward += 1
                        legal_moves_played.add(move)
                    else:
                        reward -= 1
                except:
                    reward -= 1
                    #print("failed move")
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

    def compute_loss_bis(self, new_logits, ref_logits, advantages, sequences,legal_mask):
        """Compute the loss as the ratio of π_θ(a|s) to π_θ_no_grad(a|s), minus KL(π_θ || π_ref)."""
        new_logits = new_logits[:,:,63:-1,:]
        ref_logits = ref_logits[:,:,63:-1,:]
        #print(new_logits.shape,sequences.shape)
        matches = (sequences == end_index)
        #print(matches.device,sequences.device)
        first_indices = torch.where(matches, torch.arange(sequences.shape[2],device = matches.device).expand_as(sequences), sequences.shape[2])
        first_occurrence = first_indices.min(dim=-1).values
        to_consider = first_occurrence != 192

        logits_of_move_played = new_logits.gather(dim=2, index=(first_occurrence-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1929))
        logits_of_move_played = logits_of_move_played.squeeze(2)
        #print(logits_of_move_played.shape)

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

        #kl between logits of move played and legal_mask
        legal_mask_logits = legal_mask + (1-legal_mask) * (-1000)
        legal_loss = F.kl_div(F.log_softmax(logits_of_move_played,dim=-1),F.softmax(legal_mask_logits,dim=-1),reduction = "none")
        legal_loss = legal_loss.sum(dim=-1) * to_consider
        legal_loss = legal_loss.sum() / to_consider.sum()
        
        loss = 0.1*loss + legal_loss
        print(first_occurrence,legal_loss)
        legal_prob = F.softmax(logits_of_move_played,dim=-1)
        legal_prob = legal_prob * legal_mask
        legal_prob = legal_prob.sum(dim=-1)
        legal_prob = legal_prob * to_consider
        legal_prob = legal_prob.sum() / to_consider.sum()
        #print(to_consider)
        #print(legal_prob)
        
        
        return loss, mean_kl,legal_prob

    def compute_legal_mask(self,fens):
        legal_mask = torch.zeros((len(fens),1929),dtype = torch.float32,device = "cuda")
        for i in range(len(fens)):
            board = chess.Board(fens[i])
            for move in board.legal_moves:
                if move.uci() in policy_index:
                    move_id = policy_index.index(move.uci())
                else:
                    move_id = policy_index.index(move.uci()[:-1])
                legal_mask[i][move_id] = 1
        return legal_mask.unsqueeze(0).expand(self.G,-1,-1)


    def train_step(self, batch):
        """Process one batch of board positions and target moves."""
        board_tokens_batch, target_moves_batch,fens = batch
        legal_mask = self.compute_legal_mask(fens)
        sequences, old_probs,board_tokens_batch = self.generate_sequence(board_tokens_batch)
        rewards, format_reward, move_rewards,legal_rewards,acc  = self.compute_rewards(sequences, target_moves_batch,fens)
        #print(rewards,rewards.shape)
        max_reward= move_rewards.max(dim=0)[0].mean()
        mean_reward = move_rewards.mean()
        advantages = (rewards - rewards.mean(dim=0) + self.alpha * (rewards.mean(dim=0)- rewards.mean())) / (rewards.std(dim=0) + 0.1)

        batched = sequences.view(-1,sequences.shape[2])
        new_logits, _ , _ = self.model((board_tokens_batch,batched),compute_loss=True)
        new_logits = new_logits.view(self.G,-1,new_logits.shape[1],new_logits.shape[2])
        with torch.no_grad():
            ref_logits, _ , _ = self.ref_model((board_tokens_batch,batched),compute_loss=True)
        ref_logits = ref_logits.view(self.G,-1,ref_logits.shape[1],ref_logits.shape[2])

        #print(new_logits.shape,ref_logits.shape,advantages.shape,sequences.shape)
        loss,kl,legal_prob = self.compute_loss_bis(new_logits, ref_logits, advantages,sequences,legal_mask)
        loss.backward()
        self.loss_t+= loss.item()
        self.kl_t += kl.mean().item()
        self.rewards_t += rewards.mean().item()
        self.format_reward_t += format_reward.mean().item()
        self.move_rewards_t += move_rewards.mean().item()
        self.legal_rewards_t += legal_rewards.mean().item()
        self.acc_t += acc
        self.legal_probs += legal_prob.item()
        if (self.count+1)%self.grad_acc==0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if wandb_log:
                wandb.log({"loss":self.loss_t/self.grad_acc, "kl_divergence": self.kl_t/self.grad_acc,\
                "rewards": self.rewards_t/self.grad_acc, "format_rewards":self.format_reward_t/self.grad_acc,\
                "move_rewards": self.move_rewards_t/self.grad_acc,"legal_rewards": self.legal_rewards_t/self.grad_acc, "accuracy": self.acc_t / self.grad_acc, "legal_prob": self.legal_probs/self.grad_acc,\
                "max_move_reward": max_reward.item(), "mean_move_reward": mean_reward.item()})
            '''
            self.writer.add_scalar("Loss", self.loss_t/self.grad_acc, self.count)
            self.writer.add_scalar("KL Divergence", self.kl_t/self.grad_acc, self.count)
            self.writer.add_scalar("Rewards", self.rewards_t/self.grad_acc, self.count)
            self.writer.add_scalar("Format Rewards", self.format_reward_t/self.grad_acc, self.count)
            self.writer.add_scalar("Move Rewards", self.move_rewards_t/self.grad_acc, self.count)
            self.writer.add_scalar("Accuracy", self.acc_t / self.grad_acc, self.count)
            '''
            self.loss_t = 0
            self.kl_t = 0
            self.rewards_t = 0
            self.legal_rewards_t = 0
            self.format_reward_t=0
            self.move_rewards_t=0
            self.acc_t = 0
            self.legal_probs = 0
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
    wandb_log = True
    if wandb_log:
        wandb.init(project="ChessRL-GRPO")
    from utils.parse import dir_iterator
    dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
    gen = dir_iterator(dir_path,triple = True)
    #load weights 
    #model.load_state_dict(torch.load("fine_tune/new_13000.pt"))
    #reference.load_state_dict(torch.load("fine_tune/new_13000.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    # Initialize trainer
    trainer = ChessGRPOTrainer(model, reference, optimizer)
    
    import copy

    start_think_index = policy_index.index("<thinking>")
    end_think_index = policy_index.index("</thinking>")
    end_variation_index = policy_index.index("end_variation")
    end_index = policy_index.index("end")
    
    # Train step
    num_steps = 100000
    for step in range(num_steps):
        batch = next(gen)
        #print(batch)
        loss = trainer.train_step(batch)
        if (step + 1) % (12*trainer.grad_acc) == 0:
            trainer.ref_model = copy.deepcopy(trainer.model)
        #print(loss.item())
        if (step + 1) % 100 == 0:
            trainer.stockfish.close()
            trainer.stockfish = StockfishEvaluator()
