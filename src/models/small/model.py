import torch
import sys
import os
import bulletchess
import torch.nn.functional as F
import torch.nn as nn

from .attn import RelativeMultiHeadAttention2
from .vocab import policy_index, policy_to_idx
from .fen_encoder import fen_to_tensor
from typing import List, Tuple, Union

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

        self.gelu = torch.nn.GELU().to("cuda")
    
    def forward(self,x,pos_enc):

        attn_out = self.attention(x,x,x,pos_enc)
        x = attn_out + x

        x = self.norm1(x)

        y = self.ff1(x)
        y = self.gelu(y)
        y = self.ff2(y)
        #y, loss = self.moe(x)


        y = y+x

        y = self.norm2(y)

        return y

class AbsolutePositionalEncoder(torch.nn.Module):
    def __init__(self,d_model):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(64).unsqueeze(1)
        
        # Create positional encoding tensor
        positional_encoding = torch.zeros(1, 64, d_model)
        
        _2i = torch.arange(0, d_model, step=2).float()
        
        positional_encoding[:, : , 0::2]= torch.sin(self.position / (10000 ** (_2i/ d_model)))
        
        positional_encoding[:, : , 1::2]= torch.cos(self.position / (10000 ** (_2i/ d_model)))
        
        # Register as buffer so it's properly handled during tracing/export
        self.register_buffer('positional_encoding', positional_encoding)
    
    def forward(self,x):
        batch_size = x.shape[0]

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
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Create position indices: [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_embed = self.positional_embedding(positions)  # [1, seq_len, d_model]
        
        # Expand to match batch size
        pos_embed = pos_embed.expand(batch_size, -1, -1)  # [batch_size, seq_len, d_model]

        return pos_embed

class BT4(torch.nn.Module):
    def __init__(self,num_layers = 4, d_model=256, d_ff=512, num_heads = 4):
        super().__init__()
        self.is_thinking_model = False
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
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

    def forward(self, inp, *, compute_loss: bool = False, return_all: bool = False, step = None):
        x = inp[0]
        b = x.shape[0]
        seq_len = x.shape[1]
        emb = x.shape[-1]  # Last dimension (should be 19)
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
        ## Implement how much we go through the layers here 

        
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

        dk = torch.sqrt(torch.tensor(self.d_model, dtype=matmul_qk.dtype, device=matmul_qk.device))

        policy_attn_logits = matmul_qk / dk
        policy_attn_logits = policy_attn_logits.view(b,seq_len,64*64)

        policy = self.policy_head(policy_attn_logits)#shape (b,seq_len,1929)

        #queries Q coming from board embedding
        # keys K and values V coming from policy tokens

        # perform cross attention between board embeddings and policy tokens
        # 
        policy = policy #+ self.final_decoder(policy+self.positional_policy(policy),board_embeddings+self.positional_board(board_embeddings))


        outputs = (policy, value_h, value_h_q)

        if compute_loss:
            targets = inp[1]
            true_values = inp[3]
            q_values = inp[4]
            true_values = q_values
            
            z = torch.argmax(true_values, dim=-1)
            q = torch.argmax(q_values, dim=-1)
            value_h_q_probs = torch.softmax(value_h_q, dim=-1)
            
            # Always compute policy loss
            loss_policy = F.cross_entropy(policy.view(-1,policy.size(-1)), targets.view(-1),ignore_index=1928)
            
            # Create mask for samples where true_values/q_values is not [0,0,0]
            # Check if any element in the last dimension is non-zero
            valid_mask = (true_values.sum(dim=-1) != 0) & (q_values.sum(dim=-1) != 0)
            
            # Only compute value losses if we have valid samples
            if valid_mask.any():
                # Filter to only valid samples
                valid_value_h = value_h[valid_mask]
                valid_value_h_q_probs = value_h_q_probs[valid_mask]
                valid_z = z[valid_mask]
                valid_q_values = q_values[valid_mask]
                
                loss_value = F.cross_entropy(valid_value_h.view(-1,valid_value_h.size(-1)), valid_z.view(-1))
                loss_q = F.mse_loss(valid_value_h_q_probs.view(-1,valid_value_h_q_probs.size(-1)), valid_q_values.view(-1,3))
            else:
                # No valid samples, set losses to zero
                loss_value = torch.tensor(0.0, device=value_h.device, requires_grad=True)
                loss_q = torch.tensor(0.0, device=value_h_q.device, requires_grad=True)
            
            return policy, value_h, value_h_q, loss_policy, loss_value,loss_q, targets, z,q
        if return_all:
            return outputs
        return policy

    def get_move_from_fen_no_thinking(self, fen, T = 1,device = "cuda",force_legal = True,return_probs = False, use_fp16: bool = False):
        board = bulletchess.Board.from_fen(fen)
        x = torch.from_numpy(fen_to_tensor(fen)).to(device).to(torch.float32)
        x = x.view(1,1,8,8,19)

        if use_fp16 and torch.device(device).type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = self([x.half(), None])
        else:
            logits = self([x, None])
        logits = logits.view(-1,1929)
        legal_move_mask = torch.zeros((1, 1929), device=device)
        for legal_move in board.legal_moves():
            if legal_move.uci()[-1] == 'n':
                legal_move_uci = legal_move.uci()[:-1]
            else:
                legal_move_uci = legal_move.uci()
            idx = policy_to_idx.get(legal_move_uci)
            if idx is not None:
                legal_move_mask[0][idx] = 1
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
            if return_probs:
                # top_10_probs, top_10_indices = torch.topk(probs, 10)
                # top_10_moves = [policy_index[i] for i in top_10_indices.squeeze()]
                # top_10_probs = top_10_probs.squeeze()
                # #print top 10 moves and probs
                # for move, prob in zip(top_10_moves, top_10_probs):
                #     print(f"{move}: {prob.item()}")
                return sampled, probs
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
            b = x.shape[0]
            seq_len = x.shape[1]
            emb = x.shape[-1]  # Last dimension (should be 19)
            x_processed = x.view(b * seq_len, 64, emb)
            x_processed = self.linear1(x_processed)
            x_processed = torch.nn.GELU()(x_processed)
            x_processed = self.layernorm1(x_processed)
            x_processed = self.ma_gating(x_processed)
            
            pos_enc = self.positional(x_processed)
            for i in range(self.num_layers):
                x_processed = self.layers[i](x_processed, pos_enc)
            
            value_logits = self.value_head_q(x_processed)
            value_logits = value_logits.view(b, seq_len, 3)
            value = torch.softmax(value_logits,dim=-1)
        return value.squeeze()  # Remove batch and sequence dimensions

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
            b = batch_x.shape[0]
            seq_len = batch_x.shape[1]
            emb = batch_x.shape[-1]  # Last dimension (should be 19)
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
        board = bulletchess.Board.from_fen(fen)
        
        # Determine whose turn it is
        is_white_turn = board.turn == bulletchess.WHITE
        
        legal_moves = list(board.legal_moves())
        if len(legal_moves) == 0:
            return [], torch.empty(0, device=device)
        
        # Get all resulting FENs after each move
        resulting_fens = []
        for move in legal_moves:
            board.apply(move)
            resulting_fens.append(board.fen())
            board.undo()
        
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

    def get_best_move_value(self, fen, T=1, device="cuda", return_probs=False,to_fall_back_to_policy=False):
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
        # get value of position for side to move. If value is higher than 0.9, sample from policy
        value = self.get_position_value(fen, device)
        board = bulletchess.Board.from_fen(fen)
        
        # Determine whose turn it is
        is_white_turn = board.turn == bulletchess.WHITE
        if is_white_turn:
            value = value[2]-value[0]
        else:
            value = value[0]-value[2]
        if value > 0.9 and to_fall_back_to_policy:
            self.fall_back_to_policy = True
        if to_fall_back_to_policy and self.fall_back_to_policy:
            return self.get_move_from_fen_no_thinking(fen, T, device, force_legal=True, return_probs=return_probs)

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

    def batch_get_moves_from_fens(self, fens: List[str], T: float = 0.0, device: str = None, force_legal: bool = True, return_probs: bool = False, use_fp16: bool = False) -> Union[List[str], Tuple[List[str], torch.Tensor]]:
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        if not fens:
            if return_probs:
                return [], torch.empty(0, 1929, device=device)
            return []

        # Encode all FENs to a batch tensor: (B, 1, 8, 8, 19)
        encoded_positions = []
        for fen in fens:
            x = torch.from_numpy(fen_to_tensor(fen)).to(device).to(torch.float32)
            encoded_positions.append(x)
        batch_x = torch.stack(encoded_positions, dim=0).unsqueeze(1)

        # Forward pass -> logits (B, 1929)
        self.eval()
        with torch.inference_mode():
            if use_fp16 and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = self([batch_x.half(), None])
            else:
                logits = self([batch_x, None])
        B = logits.shape[0]
        logits = logits.view(B, 1929)

        # Build legal move masks per position using Bullet Chess
        if force_legal:
            legal_mask = torch.full((B, 1929), -999.0, device=device, dtype=logits.dtype)
            for i, fen in enumerate(fens):
                board = bulletchess.Board.from_fen(fen)
                for mv in board.legal_moves():
                    u = mv.uci()
                    if u[-1] == 'n':
                        u = u[:-1]
                    idx = policy_to_idx.get(u)
                    if idx is not None:
                        legal_mask[i, idx] = 0.0
            logits = logits + legal_mask

        # Sampling per row
        moves: List[str] = []
        probs_tensor: torch.Tensor
        if T == 0.0:
            best_idx = torch.argmax(logits, dim=1)
            for idx in best_idx.tolist():
                moves.append(policy_index[idx])
            if return_probs:
                probs_tensor = torch.zeros_like(logits)
                probs_tensor[torch.arange(B), best_idx] = 1.0
        else:
            scaled = logits / T
            probs_tensor = F.softmax(scaled, dim=1)
            sampled = torch.multinomial(probs_tensor, num_samples=1).squeeze(1)
            for idx in sampled.tolist():
                moves.append(policy_index[idx])

        if return_probs:
            return moves, probs_tensor
        return moves

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
        b = x.shape[0]
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
        b = x.shape[0]
        x = self.dense1(x)
        x = torch.nn.GELU()(x)
        x = x.view(b,-1)
        x = self.dense2(x)
        x = torch.nn.GELU()(x)
        x = self.dense3(x)
        return x

def analyze_model_parameters(model):
    """Analyze and organize model parameters by component type"""
    
    # Initialize parameter counters
    input_encoding_params = 0
    attention_params = 0
    dff_params = 0
    norm_params = 0
    output_head_params = 0
    
    # Parameter details for organization
    param_details = {
        'input_encoding': [],
        'attention': [],
        'dff': [],
        'norm': [],
        'output_heads': []
    }
    
    print("\n" + "="*60)
    print("MODEL PARAMETER ANALYSIS")
    print("="*60)
    
    print("\nDetailed Parameter Breakdown:")
    print("-" * 40)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        print(f"{name}: {param_count:,}")
        
        # Categorize parameters
        if any(component in name for component in ['linear1', 'layernorm1']):
            input_encoding_params += param_count
            param_details['input_encoding'].append((name, param_count))
            
        elif 'attention' in name:
            attention_params += param_count
            param_details['attention'].append((name, param_count))
            
        elif any(component in name for component in ['ff1', 'ff2']):
            dff_params += param_count
            param_details['dff'].append((name, param_count))
            
        elif any(component in name for component in ['norm1', 'norm2']):
            norm_params += param_count
            param_details['norm'].append((name, param_count))
            
        else:  # Output heads and other components
            output_head_params += param_count
            param_details['output_heads'].append((name, param_count))
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n" + "="*60)
    print("PARAMETER ORGANIZATION BY COMPONENT")
    print("="*60)
    
    # Input Encoding
    print(f"\n1. INPUT ENCODING: {input_encoding_params:,} parameters ({input_encoding_params/total_params*100:.2f}%)")
    for name, count in param_details['input_encoding']:
        print(f"   {name}: {count:,}")
    
    # Attention Components
    print(f"\n2. ATTENTION COMPONENTS: {attention_params:,} parameters ({attention_params/total_params*100:.2f}%)")
    
    # Group by layer for attention
    attention_by_layer = {}
    for name, count in param_details['attention']:
        layer_num = name.split('.')[1] if 'layers.' in name else 'other'
        if layer_num not in attention_by_layer:
            attention_by_layer[layer_num] = []
        attention_by_layer[layer_num].append((name, count))
    
    # Calculate per-layer attention params
    layer_attention_params = 0
    if '0' in attention_by_layer:
        layer_attention_params = sum(count for _, count in attention_by_layer['0'])
        print(f"   Per Layer: {layer_attention_params:,} parameters")
        print(f"   Total Layers: {model.num_layers}")
        for name, count in attention_by_layer['0']:
            component_name = name.split('.')[-2] + '.' + name.split('.')[-1]
            print(f"     {component_name}: {count:,}")
    
    # DFF Components
    print(f"\n3. DFF (FEED-FORWARD): {dff_params:,} parameters ({dff_params/total_params*100:.2f}%)")
    
    # Group by layer for DFF
    dff_by_layer = {}
    for name, count in param_details['dff']:
        layer_num = name.split('.')[1] if 'layers.' in name else 'other'
        if layer_num not in dff_by_layer:
            dff_by_layer[layer_num] = []
        dff_by_layer[layer_num].append((name, count))
    
    # Calculate per-layer DFF params
    layer_dff_params = 0
    if '0' in dff_by_layer:
        layer_dff_params = sum(count for _, count in dff_by_layer['0'])
        print(f"   Per Layer: {layer_dff_params:,} parameters")
        for name, count in dff_by_layer['0']:
            component_name = name.split('.')[-2] + '.' + name.split('.')[-1]
            print(f"     {component_name}: {count:,}")
    
    # Layer Normalization
    print(f"\n4. LAYER NORMALIZATION: {norm_params:,} parameters ({norm_params/total_params*100:.2f}%)")
    
    # Group by layer for norms
    norm_by_layer = {}
    for name, count in param_details['norm']:
        layer_num = name.split('.')[1] if 'layers.' in name else 'other'
        if layer_num not in norm_by_layer:
            norm_by_layer[layer_num] = []
        norm_by_layer[layer_num].append((name, count))
    
    # Calculate per-layer norm params
    layer_norm_params = 0
    if '0' in norm_by_layer:
        layer_norm_params = sum(count for _, count in norm_by_layer['0'])
        print(f"   Per Layer: {layer_norm_params:,} parameters")
        for name, count in norm_by_layer['0']:
            component_name = name.split('.')[-2] + '.' + name.split('.')[-1]
            print(f"     {component_name}: {count:,}")
    
    # Output Heads
    print(f"\n5. OUTPUT HEADS: {output_head_params:,} parameters ({output_head_params/total_params*100:.2f}%)")
    
    # Organize output heads by type
    policy_head_params = 0
    value_head_params = 0
    other_output_params = 0
    
    for name, count in param_details['output_heads']:
        if 'policy' in name:
            policy_head_params += count
            print(f"   Policy - {name}: {count:,}")
        elif 'value_head' in name:
            value_head_params += count
            head_type = "Value Head Q" if "value_head_q" in name else "Value Head"
            component_name = name.split('.')[-2] + '.' + name.split('.')[-1]
            print(f"   {head_type} - {component_name}: {count:,}")
        else:
            other_output_params += count
            print(f"   Other - {name}: {count:,}")
    
    print(f"\n   Policy Head Total: {policy_head_params:,} parameters")
    print(f"   Value Heads Total: {value_head_params:,} parameters")
    if other_output_params > 0:
        print(f"   Other Output Total: {other_output_params:,} parameters")
    
    # Summary Table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Component':<25} {'Parameters':<15} {'Percentage':<12}")
    print("-" * 52)
    print(f"{'Input Encoding':<25} {input_encoding_params:<15,} {input_encoding_params/total_params*100:<12.2f}%")
    print(f"{'Attention':<25} {attention_params:<15,} {attention_params/total_params*100:<12.2f}%")
    print(f"{'DFF (Feed-Forward)':<25} {dff_params:<15,} {dff_params/total_params*100:<12.2f}%")
    print(f"{'Layer Normalization':<25} {norm_params:<15,} {norm_params/total_params*100:<12.2f}%")
    print(f"{'Output Heads':<25} {output_head_params:<15,} {output_head_params/total_params*100:<12.2f}%")
    print("-" * 52)
    print(f"{'TOTAL':<25} {total_params:<15,} {'100.00%':<12}")
    
    # Architecture Insights
    print("\n" + "="*60)
    print("ARCHITECTURE INSIGHTS")
    print("="*60)
    print(f"• Model Type: {model.num_layers}-layer Transformer")
    print(f"• Hidden Dimension: {model.d_model}")
    print(f"• Feed-Forward Dimension: {model.d_ff}")
    print(f"• Number of Attention Heads: {model.num_heads}")
    print(f"• Attention-Heavy Design: {attention_params/total_params*100:.1f}% of parameters in attention")
    print(f"• Large Output Capacity: {output_head_params/total_params*100:.1f}% in policy/value heads")
    print(f"• Efficient Normalization: {norm_params/total_params*100:.3f}% in layer norms")
    print(f"• Minimal Input Processing: {input_encoding_params/total_params*100:.3f}% in input encoding")
    
    if policy_head_params > 0:
        print(f"• Policy Head Size: {policy_head_params:,} parameters ({policy_head_params/total_params*100:.1f}%)")
    if value_head_params > 0:
        print(f"• Value Heads Size: {value_head_params:,} parameters ({value_head_params/total_params*100:.1f}%)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    model = BT4(num_layers=4,d_model=256,d_ff=512,num_heads=4).to("cuda")
    # Input tensor shape: (batch_size, sequence_length, board_dim1, board_dim2, embedding_dim)
    # For the model, this becomes (batch_size * sequence_length, board_dim1 * board_dim2, embedding_dim)
    # (4, 64, 8, 8, 19) -> (256, 64, 19)
    random_tensor = torch.randn(4, 64, 8, 8, 19).to("cuda")
    output = model([random_tensor])
    print("Model output shape:", output.shape)
    print("Model output (first element of batch):", output[0])
    
    # Analyze model parameters dynamically
    analyze_model_parameters(model)
