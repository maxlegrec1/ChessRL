import torch
#from mixture_of_experts import MoE
import sys
import os
# import numpy as np # Removed as it's no longer used
# import chess # Removed as it's no longer used
import torch.nn.functional as F
import torch.nn as nn
sys.path.append(os.getcwd())
try:
    from .attn import RelativeMultiHeadAttention2 # User changed from .attn
except:
    from attn import RelativeMultiHeadAttention2 # User changed from .attn
# from gen_TC import policy_index # User removed
num_layers = 10 # User changed
d_model = 512   # User changed
d_ff = 736    # User changed
num_heads = 8   # User changed
from utils.vocab import policy_index
from utils.fen_encoder import fen_to_tensor
import chess
class MaGating(torch.nn.Module):
    def __init__(self, d_model): # Added d_model argument
        super().__init__()
        # User had (64,1024) but d_model is now 512. Parameterizing with d_model.
        self.a = torch.nn.Parameter(torch.zeros(64, d_model)) # Use d_model
        self.b = torch.nn.Parameter(torch.ones(64, d_model))  # Use d_model

    def forward(self,x):
        return x*torch.exp(self.a) + self.b

class EncoderLayer(torch.nn.Module):
    def __init__(self,d_model,d_ff,num_heads):
        super().__init__()
        self.attention = RelativeMultiHeadAttention2(d_model,num_heads,0).to("cuda")
        self.norm1 = torch.nn.LayerNorm(d_model).to("cuda")
        self.norm2 = torch.nn.LayerNorm(d_model).to("cuda")


        self.ff1 = torch.nn.Linear(d_model,d_ff).to("cuda")
        self.ff2 = torch.nn.Linear(d_ff,d_model).to("cuda")
        '''
        self.moe = MoE(
            dim = 1024,
            num_experts = 4,
            hidden_dim = 1536,
            second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
            second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            loss_coef = 1e-2 
        )'''

        self.gelu = torch.nn.GELU().to("cuda")
    
    def forward(self,x,pos_enc):

        attn_out = self.attention(x,x,x,pos_enc)
        x = attn_out + x

        x = self.norm1(x)

        y = self.ff1(x)
        y = self.ff2(y)
        y = self.gelu(y)
        #y, loss = self.moe(x)


        y = y+x

        y = self.norm2(y)

        return y

class AbsolutePositionalEncoder(torch.nn.Module):
    def __init__(self,d_model):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(64).unsqueeze(1)
        
        self.positional_encoding = torch.zeros(1, 64, d_model).to("cuda")

        _2i = torch.arange(0, d_model, step=2).float()

        self.positional_encoding[:, : , 0::2]= torch.sin(self.position / (10000 ** (_2i/ d_model)))

        self.positional_encoding[:, : , 1::2]= torch.cos(self.position / (10000 ** (_2i/ d_model)))
    
    def forward(self,x):
        batch_size,_,_ = x.size()

        return self.positional_encoding.expand(batch_size, -1, -1)
class LearnedPositionalEncoder(nn.Module):
    def __init__(self, d_model=1929, max_len=64):
        super(LearnedPositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Learned positional embeddings
        self.positional_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # Create position indices: [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_embed = self.positional_embedding(positions)  # [1, seq_len, d_model]
        
        # Expand to match batch size
        pos_embed = pos_embed.expand(batch_size, -1, -1)  # [batch_size, seq_len, d_model]

        return pos_embed

class BT4(torch.nn.Module):
    def __init__(self,num_layers = num_layers,d_model=d_model,d_ff=d_ff,num_heads = num_heads):
        super().__init__()
        self.is_thinking_model = False
        self.d_model = d_model

        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList([EncoderLayer(d_model,d_ff,num_heads) for _ in range(num_layers)])
        #self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model,num_heads,d_ff,0,"gelu",batch_first=True),num_layers)

        self.linear1 = torch.nn.Linear(19,d_model)

        self.layernorm1 = torch.nn.LayerNorm(d_model)

        self.policy_tokens_lin = torch.nn.Linear(d_model,d_model)

        self.queries_pol = torch.nn.Linear(d_model,d_model)

        self.keys_pol = torch.nn.Linear(d_model,d_model)

        self.positional = AbsolutePositionalEncoder(d_model)

        #self.positional_board = LearnedPositionalEncoder(64*4)

        #self.positional_policy = LearnedPositionalEncoder(1929)

        self.ma_gating = MaGating(d_model) # Pass d_model

        self.policy_head = torch.nn.Linear(64*64,1929,bias=False)

        self.value_head = ValueHead(d_model)
        self.value_head_q = ValueHeadQ(d_model)
        #self.final_decoder = PolicyRefinerDecoder(1929,64*4)

    def forward(self,inp,compute_loss = False):
        x = inp[0]
        b,seq_len,_,_,emb = x.size()
        x = x.view(b*seq_len,64,emb)
        #print(x.shape)
        x = self.linear1(x)
        #add gelu
        x = torch.nn.GELU()(x)

        x = self.layernorm1(x)

        #add ma gating 
        x = self.ma_gating(x)
        pos_enc = self.positional(x)
        for i in range(self.num_layers):
            x = self.layers[i](x,pos_enc)
        value_h = self.value_head(x)
        value_h = value_h.view(b,seq_len,3)
        value_h_q = self.value_head_q(x)
        value_h_q = value_h_q.view(b,seq_len,3)
        policy_tokens = self.policy_tokens_lin(x)
        policy_tokens = torch.nn.GELU()(policy_tokens)
        policy_tokens = policy_tokens + pos_enc
        queries = self.queries_pol(policy_tokens)

        keys = self.keys_pol(policy_tokens)

        matmul_qk = torch.matmul(queries,torch.transpose(keys,-2,-1))

        dk = torch.sqrt(torch.tensor(self.d_model))

        policy_attn_logits = matmul_qk / dk
        policy_attn_logits = policy_attn_logits.view(b,seq_len,64*64)

        policy = self.policy_head(policy_attn_logits)#shape (b,seq_len,1929)

        #queries Q coming from board embedding
        # keys K and values V coming from policy tokens

        # perform cross attention between board embeddings and policy tokens
        # 
        policy = policy #+ self.final_decoder(policy+self.positional_policy(policy),board_embeddings+self.positional_board(board_embeddings))


        if compute_loss:
            targets = inp[1]
            true_values = inp[3]
            q_values = inp[4]
            #print(true_values[:5],q_values[:5])
            loss_policy = F.cross_entropy(policy.view(-1,policy.size(-1)), targets.view(-1),ignore_index=1928)
            z = torch.argmax(true_values, dim=-1)
            loss_value = F.cross_entropy(value_h.view(-1,value_h.size(-1)), z.view(-1),ignore_index=3)
            #value_q = q_values[:,0]-q_values[:,2]
            loss_q = F.mse_loss(value_h_q.view(-1,value_h_q.size(-1)), q_values.view(-1,3))
            return policy, value_h, loss_policy, loss_value,loss_q, targets, z
        return policy

    def get_move_from_fen_no_thinking(self, fen, T = 1,device = "cuda",force_legal = True,return_probs = False):
        board = chess.Board()
        board.set_fen(fen)
        x = torch.from_numpy(fen_to_tensor(fen)).to("cuda").to(torch.float32)
        x = x.view(1,1,8,8,19)

        logits= self([x,None])
        logits = logits.view(-1,1929)
        legal_move_mask = torch.zeros((1, 1929), device=device)
        for legal_move in board.legal_moves:
            if legal_move.uci()[-1] == 'n':
                legal_move_uci = legal_move.uci()[:-1]
            else:
                legal_move_uci = legal_move.uci()
            legal_move_mask[0][policy_index.index(legal_move_uci)] = 1
        #set all illegal moves to -inf
        if force_legal:
            logits = logits + (1-legal_move_mask) * -999

        if T == 0:
            #print("using argmax")
            sampled = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits/T, dim=-1)
            #sample a move according to the probabilities
            #print(probs.shape)
            sampled = torch.multinomial(probs, num_samples=1)
        move = policy_index[sampled.item()]

        return move

    def get_position_value(self, fen, device="cuda"):
        """
        Get the value evaluation for a given FEN position.
        Returns the value vector [black_win_prob, draw_prob, white_win_prob]
        """
        x = torch.from_numpy(fen_to_tensor(fen)).to(device).to(torch.float32)
        x = x.view(1, 1, 8, 8, 19)
        
        # Forward pass through the model to get value
        with torch.no_grad():
            # We need to run through the model layers to get to value_head
            b, seq_len, _, _, emb = x.size()
            x_processed = x.view(b * seq_len, 64, emb)
            x_processed = self.linear1(x_processed)
            x_processed = torch.nn.GELU()(x_processed)
            x_processed = self.layernorm1(x_processed)
            x_processed = self.ma_gating(x_processed)
            
            pos_enc = self.positional(x_processed)
            for i in range(self.num_layers):
                x_processed = self.layers[i](x_processed, pos_enc)
            
            value_logits = self.value_head_q(x_processed)
            value_logits = value_logits.view(b, seq_len, 1)
             
        return value_logits.squeeze()  # Remove batch and sequence dimensions

    def get_batch_position_values(self, fens, device="cuda"):
        """
        Get the value evaluation for a batch of FEN positions efficiently.
        Args:
            fens: List of FEN strings
            device: Device to run computations on
        Returns:
            value_probs: Tensor of shape [batch_size, 3] with [black_win_prob, draw_prob, white_win_prob] for each position
        """
        if len(fens) == 0:
            return torch.empty(0, 3, device=device)
        
        # Convert all FENs to tensors and stack them
        position_tensors = []
        for fen in fens:
            x = torch.from_numpy(fen_to_tensor(fen)).to(device).to(torch.float32)
            position_tensors.append(x)
        
        # Stack to create batch: [batch_size, 8, 8, 19]
        batch_x = torch.stack(position_tensors, dim=0)
        # Reshape to [batch_size, 1, 8, 8, 19] for the model
        batch_x = batch_x.unsqueeze(1)
        
        # Forward pass through the model to get values
        with torch.no_grad():
            b, seq_len, _, _, emb = batch_x.size()
            x_processed = batch_x.view(b * seq_len, 64, emb)
            x_processed = self.linear1(x_processed)
            x_processed = torch.nn.GELU()(x_processed)
            x_processed = self.layernorm1(x_processed)
            x_processed = self.ma_gating(x_processed)
            
            pos_enc = self.positional(x_processed)
            for i in range(self.num_layers):
                x_processed = self.layers[i](x_processed, pos_enc)
            
            value_logits = self.value_head_q(x_processed)
            value_logits = value_logits.view(b, seq_len, 3)
            value_logits = torch.softmax(value_logits,dim=-1)
        return value_logits.squeeze(1)  # Remove sequence dimension, keep batch dimension

    def calculate_move_values(self, fen, device="cuda"):
        """
        Calculate the value for each legal move from the given position efficiently using batching.
        For white to move, value = 1/2 * draw_prob + white_win_prob
        For black to move, value = 1/2 * draw_prob + black_win_prob
        """
        board = chess.Board()
        board.set_fen(fen)
        
        # Determine whose turn it is
        is_white_turn = board.turn == chess.WHITE
        
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            return [], torch.empty(0, device=device)
        
        # Get all resulting FENs after each move
        resulting_fens = []
        for move in legal_moves:
            board.push(move)
            resulting_fens.append(board.fen())
            board.pop()
        
        # Batch process all positions in a single inference
        batch_value_q = self.get_batch_position_values(resulting_fens, device)
        
        # Calculate values from the current player's perspective
        # batch_value_probs[:, 0] = black_win_prob, [:, 1] = draw_prob, [:, 2] = white_win_prob
        batch_value_q = batch_value_q[:,2]-batch_value_q[:,0]
        if is_white_turn:
            # White's perspective: 1/2 * draw_prob + white_win_prob
            player_values = batch_value_q
        else:
            # Black's perspective: 1/2 * draw_prob + black_win_prob
            player_values = -batch_value_q
        
        return legal_moves, player_values

    def get_best_move_value(self, fen, T=1, device="cuda", return_probs=False):
        """
        Determine the best move based on the value of resulting positions using efficient batching.
        
        Args:
            fen: FEN string of the position (works for both white and black to move)
            T: Temperature for sampling (T=0 for greedy, T>0 for stochastic)
            device: Device to run computations on
            return_probs: Whether to return the probability distribution
            
        Returns:
            move: UCI string of the selected move
            probs (optional): probability distribution over moves if return_probs=True
        """
        legal_moves, move_values = self.calculate_move_values(fen, device)
        
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available")
        
        if T == 0:
            # Greedy selection - choose move with highest value
            best_idx = torch.argmax(move_values)
            selected_move = legal_moves[best_idx]
        else:
            # Stochastic selection based on move values
            # Convert values to probabilities using softmax with temperature
            probs = F.softmax(move_values / T, dim=0)
            
            # Sample according to probabilities
            sampled_idx = torch.multinomial(probs, num_samples=1)
            selected_move = legal_moves[sampled_idx.item()]
        
        # Convert chess.Move to UCI string
        move_uci = selected_move.uci()
        
        if return_probs:
            if T == 0:
                # Create one-hot distribution for greedy case
                probs = torch.zeros_like(move_values)
                probs[best_idx] = 1.0
            else:
                probs = F.softmax(move_values / T, dim=0)
            return move_uci, probs.cpu().numpy()
        
        return move_uci

class ValueHead(torch.nn.Module):
    """
    embedded_val = tf.keras.layers.Dense(self.val_embedding_size, kernel_initializer="glorot_normal",
                                                 activation=self.DEFAULT_ACTIVATION,
                                                 name=name+"/embedding")(flow)

            h_val_flat = tf.keras.layers.Flatten()(embedded_val)
            h_fc2 = tf.keras.layers.Dense(128,
                                          kernel_initializer="glorot_normal",
                                          activation=self.DEFAULT_ACTIVATION,
                                          name=name+"/dense1")(h_val_flat)

            # WDL head
            if wdl:
                value = tf.keras.layers.Dense(3,
                                              kernel_initializer="glorot_normal",
                                              name=name+"/dense2")(h_fc2)
            
    """
    def __init__(self,d_model):
        super().__init__()
        self.dense1 = torch.nn.Linear(d_model,128)
        self.dense2 = torch.nn.Linear(128*64,128)
        self.dense3 = torch.nn.Linear(128,3)

    def forward(self,x):
        b,_,_ = x.size()
        x = self.dense1(x)
        x = torch.nn.GELU()(x)
        x = x.view(b,-1)
        x = self.dense2(x)
        x = torch.nn.GELU()(x)
        x = self.dense3(x)
        return x
    
class ValueHeadQ(torch.nn.Module):
    """
    embedded_val = tf.keras.layers.Dense(self.val_embedding_size, kernel_initializer="glorot_normal",
                                                 activation=self.DEFAULT_ACTIVATION,
                                                 name=name+"/embedding")(flow)

            h_val_flat = tf.keras.layers.Flatten()(embedded_val)
            h_fc2 = tf.keras.layers.Dense(128,
                                          kernel_initializer="glorot_normal",
                                          activation=self.DEFAULT_ACTIVATION,
                                          name=name+"/dense1")(h_val_flat)

            # WDL head
            if wdl:
                value = tf.keras.layers.Dense(3,
                                              kernel_initializer="glorot_normal",
                                              name=name+"/dense2")(h_fc2)
            
    """
    def __init__(self,d_model):
        super().__init__()
        self.dense1 = torch.nn.Linear(d_model,128)
        self.dense2 = torch.nn.Linear(128*64,128)
        self.dense3 = torch.nn.Linear(128,3)

    def forward(self,x):
        b,_,_ = x.size()
        x = self.dense1(x)
        x = torch.nn.GELU()(x)
        x = x.view(b,-1)
        x = self.dense2(x)
        x = torch.nn.GELU()(x)
        x = self.dense3(x)
        return x

if __name__ == "__main__":
    model = BT4().to("cuda")
    # Input tensor shape: (batch_size, sequence_length, board_dim1, board_dim2, embedding_dim)
    # For the model, this becomes (batch_size * sequence_length, board_dim1 * board_dim2, embedding_dim)
    # (4, 64, 8, 8, 19) -> (256, 64, 19)
    random_tensor = torch.randn(4, 64, 8, 8, 19).to("cuda")
    output = model(random_tensor)
    print("Model output shape:", output.shape)
    print("Model output (first element of batch):", output[0])

