import torch
import torch.nn as nn
import math
import chess
import numpy as np
from typing import List, Tuple
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

        # 1. Input Processing Layers
        self.token_embedding = nn.Embedding(self.d_policy, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, dropout_p, max_len=N)

        # 2. Core Transformer Decoder
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout_p,
                batch_first=True,
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
    
    def get_move_from_fen(self, fen, T = 1,device = "cuda",force_legal = True,return_probs = False,return_cot = False):
        board = chess.Board(fen)
        fen_array = [fen]
        fen_emb = self.fen_encoder(fen_array)
        states = torch.from_numpy(fen_to_tensor(fen)).to(device).view(1,1,8,8,19)
        states_policy,states_board = self.encoder([states])
        states_board = states_board[:,:,:,:(self.d_model//64)].contiguous().view(states_board.shape[0],states_board.shape[1],-1) #(B,N,d_model)
        #shape (1,1,8,8,19)
        tokens = fen_emb.unsqueeze(0)
        cot_tokens = []

        transformer_input_emb = tokens
        while True:
            state_context_projected = self.policy_projection(states_policy) + states_board
            transformer_input = self.positional_encoding(transformer_input_emb)
            current_seq_representation = transformer_input
            causal_mask = nn.Transformer.generate_square_subsequent_mask(transformer_input_emb.shape[1]).to(device)
            for i, decoder_layer in enumerate(self.decoder_layers):
                seq_context_layer = decoder_layer(
                    tgt=current_seq_representation,
                    memory=current_seq_representation,
                    tgt_mask=causal_mask,
                    memory_mask=causal_mask
                )
                fused_context = self.fusion_gates[i](seq_context_layer, state_context_projected)
                fused_context = self.fusion_layernorms[i](fused_context)
                current_seq_representation = fused_context
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
            '''
            indexes = np.argsort(-probs.cpu().detach().numpy())
            if len(cot_tokens)>0 and(cot_tokens[-1] == "end_variation" or cot_tokens[-1] == "<thinking>"):
                print(cot_tokens)
                for index in indexes[:10]:
                    print(policy_index[index],probs[index].item())
                print("--------------------------------")'''
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
        try:
            move = cot_tokens[-2]
        except:
            move = cot_tokens[-1]
        if return_cot:
            return move,cot_tokens
        else:
            return move  

if __name__ == "__main__":
    """
    Main function to initialize, test, and inspect the ReasoningTransformer.
    """
    model = ReasoningTransformer()
    model.load_state_dict(torch.load("fine_tune/fine_tune_paradigm_80001.pt"))
    model.to("cuda")

    
    fen = "6K1/4k1P1/8/8/8/7r/8/5R2 w - - 0 1"


    move,cot = model.get_move_from_fen(fen,T=0.05,return_cot=True)
    print(move,cot)
    