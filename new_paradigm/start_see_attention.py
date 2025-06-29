import torch
import torch.nn as nn
import math
import chess
import numpy as np
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from .model_v4 import BT4
except:
    from model_v4 import BT4
# Helper Modules
from utils.fen_encoder import FenEncoderMultiple,fen_to_tensor
from utils.vocab import policy_index
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, dropout_p: float = 0.0, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # Changed to [1, max_len, d_model] for batch_first
        pe[0, :, 0::2] = torch.sin(position * div_term).squeeze(1)
        pe[0, :, 1::2] = torch.cos(position * div_term).squeeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]  # Use x.size(1) for seq_len with batch_first
        return self.dropout(x)

class GatedFusion(nn.Module):
    """
    Fuses two tensors using a learned, context-dependent gate.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # This layer computes the gate from the combined contexts
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_context: torch.Tensor, state_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_context: Output from the Transformer, shape (B, N, D_model)
            state_context: Projected output from the external policy encoder, shape (B, N, D_model)
        """
        # Concatenate along the feature dimension
        combined = torch.cat((seq_context, state_context), dim=-1)
        
        # Compute the gate
        gate = self.sigmoid(self.gate_linear(combined))
        
        # Apply the gate
        fused_output = gate * seq_context + (1 - gate) * state_context
        return fused_output

class CustomTransformerDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer that exposes attention weights.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
        # Store attention weights
        self.self_attn_weights = None
        self.cross_attn_weights = None
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask, 
                                                average_attn_weights=False)
        self.self_attn_weights = self_attn_weights
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention (memory attention)
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                      key_padding_mask=memory_key_padding_mask,
                                                      average_attn_weights=False)
        self.cross_attn_weights = cross_attn_weights
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

# Main Model

class ReasoningTransformer(nn.Module):
    """
    State-Aware Reasoning Transformer (START).

    This model predicts sequences of chess moves and reasoning tokens autoregressively.
    It fuses sequential context from a Transformer with state-based context from a
    pre-trained chess model at each step.
    """
    def __init__(self, 
                 d_policy: int = 1929, 
                 d_model: int = 1024, 
                 n_layers: int = 4, 
                 n_heads: int = 8, 
                 d_ff: int = 4096, 
                 dropout_p: float = 0, 
                 N: int = 64):
        """
        Initializes the model layers.

        Args:
            d_policy (int): The vocabulary size (number of moves + special tokens).
            d_model (int): The main hidden dimension of the Transformer.
            n_layers (int): The number of Transformer decoder layers.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimension of the feed-forward network in Transformer layers.
            dropout_p (float): The dropout probability.
            N (int): The fixed sequence length.
        """
        super().__init__()
        self.N = N
        self.encoder = BT4()
        self.encoder.load_state_dict(torch.load("pretrain/v3_raw.pt"))
        self.encoder = self.encoder.to("cuda")
        #freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False


        self.fen_encoder = FenEncoderMultiple(d_model)
        self.d_policy = d_policy
        self.d_model = d_model
        self.N = N
        self.n_heads = n_heads

        # 1. Input Processing Layers
        self.token_embedding = nn.Embedding(self.d_policy, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, dropout_p, max_len=N)

        # 2. Core Transformer Decoder - Use custom decoder layers
        self.decoder_layers = nn.ModuleList([
            CustomTransformerDecoderLayer(
                d_model=self.d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout_p,
                activation='gelu'
            ) for _ in range(n_layers)
        ])

        # 3. State-Context Fusion Layers
        # This projects the policy vector from the external model into our d_model space
        self.policy_projection = nn.Linear(self.d_policy, self.d_model)
        # Original:
        # self.fusion_gate = GatedFusion(self.d_model)
        # self.fusion_layernorm = nn.LayerNorm(self.d_model)

        # New: ModuleList of fusion gates and layernorms
        self.fusion_gates = nn.ModuleList([GatedFusion(self.d_model) for _ in range(n_layers)])
        self.fusion_layernorms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(n_layers)])
        
        # 4. Final Output Head
        self.output_head = nn.Linear(self.d_model, self.d_policy)

        # 5. Causal Mask
        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(N)


    def forward(self, inp: Tuple,compute_loss = False) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            inp (Tuple): A tuple containing:
                - inp[0] (states): Precomputed board states. Shape: (B, N, *C)
                - inp[1] (tokens): Target tokens. Shape: (B, N)
                - inp[2] (fens): List of FEN strings for initial positions. Length: B

        Returns:
            torch.Tensor: Logits over the vocabulary. Shape: (B, N, P)
        """
        states, tokens, fens = inp
        B = tokens.shape[0]
        device = tokens.device
        
        # Ensure the causal mask is on the correct device
        self.causal_mask = self.causal_mask.to(device)

        # --- 1. Prepare Transformer Input Sequence ---
        
        # The FEN provides the context for the very first token in the sequence.
        # `self.fen_encoder` is an external module provided by the user.
        fen_emb = self.fen_encoder(fens) # (B, d_model)
        # For autoregressive prediction, the input is the target sequence shifted right.
        # We drop the last token to keep the length at N-1.
        input_tokens = tokens[:, :-1]
        token_embs = self.token_embedding(input_tokens) # (B, N-1, d_model)
        
        # Prepend the FEN embedding to the token embeddings.
        # This creates the full input sequence of length N for the Transformer.
        # Fen_emb serves as the "start of sequence" context.
        transformer_input_emb = torch.cat([fen_emb.unsqueeze(1), token_embs], dim=1) # (B, N, d_model)
        
        # Add positional encodings
        # Note: Pytorch Transformer layers expect (seq_len, batch, dim) by default,
        # but we use batch_first=True, so we keep it as (B, N, D).
        transformer_input = self.positional_encoding(transformer_input_emb)

        # --- 2. Get Sequential Context (Iteratively with Fusion) ---
        
        # `self.encoder` is the user's pre-trained model.
        # It is given the precomputed states for each token in the sequence.
        state_policy,state_board = self.encoder([states]) # (B, N, d_policy) (B,N,d_model)
        state_board = state_board[:,:,:,:(self.d_model//64)].contiguous().view(state_board.shape[0],state_board.shape[1],-1) #(B,N,d_model)
        # Project the policy vector into the Transformer's hidden dimension.
        # This state_context is used at each layer for fusion.
        state_context_projected = self.policy_projection(state_policy) + state_board# (B, N, d_model)

        current_seq_representation = transformer_input
        for i, decoder_layer in enumerate(self.decoder_layers):
            # --- Get Sequential Context for the current layer ---
            # The transformer decoder layer processes the sequence.
            # The causal mask ensures that the prediction for position `i` only
            # depends on the known outputs at positions less than `i`.
            # Note: The TransformerDecoderLayer expects target and memory. For self-attention in a decoder-only
            # setup, memory is typically the target itself.
            seq_context_layer = decoder_layer(
                tgt=current_seq_representation,
                memory=current_seq_representation, # Self-attention
                tgt_mask=self.causal_mask,
                memory_mask=self.causal_mask # Mask for memory if it's self-attention
            ) # (B, N, d_model)

            # --- Fuse Contexts for the current layer ---
            # Combine the sequential and state-based information using the learned gate.
            fused_context = self.fusion_gates[i](seq_context_layer, state_context_projected) # (B, N, d_model)
            fused_context = self.fusion_layernorms[i](fused_context) # Apply LayerNorm for stability
            
            current_seq_representation = fused_context # Output of fusion becomes input to next layer

        # The final fused context after all layers is current_seq_representation
        final_fused_context = current_seq_representation
        
        # --- 5. Generate Final Output ---
        
        # Project the fused representation to the vocabulary to get the final logits.
        logits = self.output_head(final_fused_context) # (B, N, d_policy)

        if compute_loss:
            targets = inp[1]
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1),ignore_index=1928)
            return logits, loss, targets
        
        return logits
    
    def get_move_from_fen(self, fen, T = 1,device = "cuda",force_legal = True,return_probs = False,return_cot = False, save_attention_maps = False):
        board = chess.Board(fen)
        fen_array = [fen]
        fen_emb = self.fen_encoder(fen_array)
        states = torch.from_numpy(fen_to_tensor(fen)).to(device).view(1,1,8,8,19)
        states_policy,states_board = self.encoder([states])
        states_board = states_board[:,:,:,:(self.d_model//64)].contiguous().view(states_board.shape[0],states_board.shape[1],-1) #(B,N,d_model)
        #shape (1,1,8,8,19)
        tokens = fen_emb.unsqueeze(0)
        cot_tokens = []
        all_attention_maps = []  # Store attention maps for all steps

        transformer_input_emb = tokens
        step = 0
        move = None
        if return_probs:
            move_probs = {}
        while True:
            state_context_projected = self.policy_projection(states_policy) + states_board
            transformer_input = self.positional_encoding(transformer_input_emb)
            current_seq_representation = transformer_input
            causal_mask = nn.Transformer.generate_square_subsequent_mask(transformer_input_emb.shape[1]).to(device)
            
            step_attention_maps = []  # Store attention maps for this step
            
            for i, decoder_layer in enumerate(self.decoder_layers):
                seq_context_layer = decoder_layer(
                    tgt=current_seq_representation,
                    memory=current_seq_representation,
                    tgt_mask=causal_mask,
                    memory_mask=causal_mask
                )
                
                # Capture attention weights if requested
                if save_attention_maps:
                    layer_attention_data = {
                        'layer': i,
                        'self_attn_weights': decoder_layer.self_attn_weights.detach().cpu() if decoder_layer.self_attn_weights is not None else None,
                        'cross_attn_weights': decoder_layer.cross_attn_weights.detach().cpu() if decoder_layer.cross_attn_weights is not None else None,
                        'sequence_length': transformer_input_emb.shape[1],
                        'tokens_so_far': cot_tokens.copy()
                    }
                    step_attention_maps.append(layer_attention_data)
                
                fused_context = self.fusion_gates[i](seq_context_layer, state_context_projected)
                fused_context = self.fusion_layernorms[i](fused_context)
                current_seq_representation = fused_context
                
            if save_attention_maps and len(cot_tokens) > 2 and cot_tokens[-2] == "</thinking>":
                all_attention_maps.append({
                    'step': step,
                    'layers': step_attention_maps,
                    'tokens_so_far': cot_tokens.copy()
                })
            
            logits = self.output_head(current_seq_representation)
            logits = logits[0,-1,:]
            legal_move_mask = torch.zeros((1929), device=device)
            for legal_move in board.legal_moves:
                if legal_move.uci()[-1] == 'n':
                    continue # disallow knight promotions for now
                else:
                    legal_move_uci = legal_move.uci()
                legal_move_mask[policy_index.index(legal_move_uci)] = 1
            #allow special tokens by default
            legal_move_mask[1927] = 1 #end token
            legal_move_mask[1926] = 1 #end_variation token
            legal_move_mask[1925] = 1 #</thinking> token
            legal_move_mask[1924] = 1 #<thinking> token

            #mask out illegal tokens
            logits = logits + (1-legal_move_mask) * -999
            probs = F.softmax(logits/T, dim=-1)
            #print probabilities if last token was end variation or start thinking

            if return_probs and len(cot_tokens) > 1 and (cot_tokens[-1] == "end_variation" or cot_tokens[-1] == "</thinking>" or cot_tokens[-1] == "<thinking>"):
                indexes = torch.argsort(probs,dim=-1,descending=True)
                move_probs[len(cot_tokens)] = (indexes[:10].tolist(),[probs[i].item() for i in indexes[:10]],[policy_index[i] for i in indexes[:10]])
            sampled = torch.multinomial(probs, num_samples=1)

            move,move_id = policy_index[sampled.item()],sampled
            cot_tokens.append(move)
            #another exit condition is cot_token length is 64 tokens
            if len(cot_tokens) >= self.N:
                break
            if move == "end":
                break
            token_emb = self.token_embedding(move_id).unsqueeze(0)
            transformer_input_emb = torch.cat([transformer_input_emb,token_emb],dim=1)
            if move == "<thinking>":
                state = -torch.ones(1,1,8,8,19).to(device)
                board.set_fen(fen)
            elif move == "</thinking>":
                state = torch.ones(1,1,8,8,19).to(device)
                board.set_fen(fen)
            elif move == "end_variation":
                state = torch.zeros(1,1,8,8,19).to(device)
                board.set_fen(fen)
            else:
                board.push_uci(move)
                current_fen_tensor = torch.from_numpy(fen_to_tensor(board.fen())).to(device)
                state = current_fen_tensor.view(1,1,8,8,19)
            state_policy,state_board = self.encoder([state])
            state_board = state_board[:,:,:,:(self.d_model//64)].contiguous().view(state_board.shape[0],state_board.shape[1],-1) #(B,N,d_model)
            states_policy = torch.cat([states_policy,state_policy],dim=1)
            states_board = torch.cat([states_board,state_board],dim=1)
            step += 1
            
        # Save attention maps if requested
        self.save_attention_visualizations(all_attention_maps, cot_tokens, fen)
        print("saving attention maps")
        try:
            move = cot_tokens[-2]
        except:
            move = cot_tokens[-1]
        if return_cot:
            return move,cot_tokens
        elif return_probs:
            return move,cot_tokens,move_probs
        else:
            return move
    
    def save_attention_visualizations(self, all_attention_maps, cot_tokens, fen):
        """
        Save attention maps as heatmaps with token names.
        """
        # Create directory if it doesn't exist
        os.makedirs("new_paradigm/attention_maps", exist_ok=True)
        
        # Create a safe filename from FEN
        safe_fen = fen.replace("/", "_").replace(" ", "_")
        
        for step_data in all_attention_maps:
            step = step_data['step']
            tokens_so_far = ['[FEN]'] + step_data['tokens_so_far']
            
            for layer_data in step_data['layers']:
                layer_idx = layer_data['layer']
                self_attn = layer_data['self_attn_weights']
                
                if self_attn is not None:
                    # self_attn shape: (batch=1, num_heads, seq_len, seq_len)
                    seq_len = self_attn.shape[-1]
                    
                    # Create token labels (truncate if too long)
                    token_labels = tokens_so_far[:seq_len]
                    if len(token_labels) < seq_len:
                        token_labels.extend([f'tok_{i}' for i in range(len(token_labels), seq_len)])
                    
                    for head in range(self_attn.shape[1]):
                        # Extract attention weights for this head
                        attn_weights = self_attn[0, head, :, :].numpy()
                        
                        # Create heatmap
                        plt.figure(figsize=(12, 10))
                        sns.heatmap(attn_weights, 
                                  xticklabels=token_labels,
                                  yticklabels=token_labels,
                                  cmap='Blues',
                                  cbar=True,
                                  square=True)
                        
                        plt.title(f'Attention Map - Step {step}, Layer {layer_idx}, Head {head}')
                        plt.xlabel('Key Tokens')
                        plt.ylabel('Query Tokens')
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        
                        # Save the plot
                        filename = f"attention_step{step:02d}_layer{layer_idx:02d}_head{head:02d}_{safe_fen[:20]}.png"
                        filepath = os.path.join("new_paradigm/attention_maps", filename)
                        plt.savefig(filepath, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Saved attention map: {filename}")

if __name__ == "__main__":
    """
    Main function to initialize, test, and inspect the ReasoningTransformer.
    """
    model = ReasoningTransformer()
    model.load_state_dict(torch.load("fine_tune/V4_100001.pt"))
    model.to("cuda")

    
    fen = "2r1nrk1/2q1b3/p2p2Pp/3Pp3/1p6/5P1B/PPPQ4/2KR3R w - - 0 22"

    # Enable attention map saving
    move, cot,probs = model.get_move_from_fen(fen, T=0.05, return_cot=False,return_probs=True, save_attention_maps=True)
    #print(move, cot)
    print(move,cot,probs)