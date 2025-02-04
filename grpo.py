import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data.vocab import policy_index


class ChessGRPOTrainer:
    def __init__(self, model, ref_model, optimizer, epsilon=0.2, beta=0.001, G=10,grad_acc = 5):
        self.model = model  # Current policy (π_θ)
        self.ref_model = ref_model  # Reference model (π_ref)
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.beta = beta
        self.G = G  # Group size
        self.epsilon = 1e-5
        self.grad_acc =  grad_acc
        self.count = 0
    def generate_sequence(self, board_tensor):
        """Generate a sequence using the current policy."""
        sequences = []
        old_probs = []
        board_expand = board_tensor.unsqueeze(0).expand(self.G,-1,-1,-1,-1).contiguous()
        board_expand = board_expand.view(self.G*board_expand.shape[1],board_expand.shape[2],board_expand.shape[3],board_expand.shape[4])
        sequences,old_probs = self.model.generate_sequence(board_expand)
        return sequences.view(self.G,-1,sequences.shape[1]),old_probs.view(self.G,-1,old_probs.shape[1],old_probs.shape[2]),board_expand

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
                    last_occurence[move] = k
                    if move in occurences:
                        occurences[move] += 1
                    else:
                        occurences[move] = 1
                        if move == end_index:
                            break
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
                # if <thinking> is played before </thinking>
                if start_think_index in occurences and end_think_index in occurences and occurences[start_think_index] == 1 and  occurences[end_think_index] == 1 and last_occurence[end_think_index] > last_occurence[start_think_index]:

                    format_reward += 2
                #if the correct move is played
                if target_move in occurences:

                    move_reward += 5
                #if the target move is played after the last think
                if end_think_index in occurences and occurences[end_think_index] == 1 and target_move in last_occurence and last_occurence[target_move] > last_occurence[end_think_index]: 
                    move_reward += 10
                    
                #if </thinking> and end are played exactly once and are separated by just one move
                if end_think_index in occurences and end_index in occurences and occurences[end_think_index] == 1 and occurences[end_index] == 1 and last_occurence[end_think_index] + 2 == last_occurence[end_index]:
                    format_reward += 10
                format_rewards[i,j] = format_reward
                move_rewards[i,j] = move_reward
        rewards = format_rewards + move_rewards
        return rewards, format_rewards, move_rewards
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




    def train_step(self, batch):
        """Process one batch of board positions and target moves."""
        board_tokens_batch, target_moves_batch = batch
        sequences, old_probs,board_tokens_batch = self.generate_sequence(board_tokens_batch)
        rewards, format_reward, move_rewards  = self.compute_rewards(sequences, target_moves_batch)
        #print(rewards,rewards.shape)
        advantages = rewards - rewards.mean(dim=0) / (rewards.std(dim=0) + 0.1)
        #print(rewards.std(dim=0))
        #print(advantages.max(),advantages.min())
        batched = sequences.view(-1,sequences.shape[2])
        new_logits, _ , _ = self.model((board_tokens_batch,batched),compute_loss=True)
        new_logits = new_logits.view(self.G,-1,new_logits.shape[1],new_logits.shape[2])
        with torch.no_grad():
            ref_logits, _ , _ = self.ref_model((board_tokens_batch,batched),compute_loss=True)
        ref_logits = ref_logits.view(self.G,-1,ref_logits.shape[1],ref_logits.shape[2])

        #print(new_logits.shape,ref_logits.shape,advantages.shape,sequences.shape)
        loss,kl = self.compute_loss(new_logits, ref_logits, advantages,sequences)
        loss.backward()
        if self.count%self.grad_acc==0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        wandb.log({"loss": loss.item(), "kl_divergence": kl.mean().item(), "rewards": rewards.mean().item(), "format_rewards": format_reward.mean().item(), "move_rewards": move_rewards.mean().item()})
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
    gen = dir_iterator(dir_path)
    #load weights 
    model.load_state_dict(torch.load("checkpoint_step_fine_tune_701.pt"))
    reference.load_state_dict(torch.load("checkpoint_step_fine_tune_701.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    # Initialize trainer
    trainer = ChessGRPOTrainer(model, reference, optimizer)
    
    import copy

    start_think_index = policy_index.index("<thinking>")
    end_think_index = policy_index.index("</thinking>")
    end_variation_index = policy_index.index("end_variation")
    end_index = policy_index.index("end")
    # Train step
    num_steps = 5000
    for _ in range(num_steps):
        batch = next(gen)
        #print(batch)
        loss = trainer.train_step(batch)
        if _ % 1000 == 0:
            trainer.ref_model = copy.deepcopy(trainer.model)
        #print(loss.item())
 