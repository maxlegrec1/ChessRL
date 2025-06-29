import torch
import torch.nn as nn
import math
from typing import List, Tuple
try:
    from .model_v3_raw import BT4
except:
    from model_v3_raw import BT4
# Helper Modules
from utils.fen_encoder import FenEncoderMultiple
import torch.nn.functional as F
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, dropout_p: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
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
        self.encoder = BT4()
        self.encoder.load_state_dict(torch.load("pretrain/v3_raw.pt"))
        self.encoder = self.encoder.to("cuda")

        self.fen_encoder = FenEncoderMultiple(d_model)
        self.d_policy = d_policy
        self.d_model = d_model
        self.N = N

        # 1. Input Processing Layers
        self.token_embedding = nn.Embedding(self.d_policy, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, dropout_p, max_len=N)

        # 2. Core Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout_p,
            batch_first=True, # Important for (B, N, D) input shape
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 3. State-Context Fusion Layers
        # This projects the policy vector from the external model into our d_model space
        self.policy_projection = nn.Linear(self.d_policy, self.d_model)
        self.fusion_gate = GatedFusion(self.d_model)
        self.fusion_layernorm = nn.LayerNorm(self.d_model)
        
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

        # --- 2. Get Sequential Context ---
        
        # The transformer decoder processes the entire sequence at once.
        # The causal mask ensures that the prediction for position `i` only
        # depends on the known outputs at positions less than `i`.
        # Note: The TransformerDecoder expects target and memory to be the same for self-attention.
        seq_context = self.transformer_decoder(
            tgt=transformer_input, 
            memory=transformer_input, 
            tgt_mask=self.causal_mask,
            memory_mask=self.causal_mask
        ) # (B, N, d_model)

        # --- 3. Get State-Based Context ---
        
        # `self.encoder` is the user's pre-trained model.
        # It is given the precomputed states for each token in the sequence.
        state_policy = self.encoder([states]) # (B, N, d_policy)
        
        # Project the policy vector into the Transformer's hidden dimension.
        state_context = self.policy_projection(state_policy) # (B, N, d_model)
        
        # --- 4. Fuse Contexts ---
        
        # Combine the sequential and state-based information using the learned gate.
        fused_context = self.fusion_gate(seq_context, state_context) # (B, N, d_model)
        fused_context = self.fusion_layernorm(fused_context) # Apply LayerNorm for stability

        # --- 5. Generate Final Output ---
        
        # Project the fused representation to the vocabulary to get the final logits.
        logits = self.output_head(fused_context) # (B, N, d_policy)

        if compute_loss:
            targets = inp[1]
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1),ignore_index=1928)
            return logits, loss, targets
        
        return logits
    

if __name__ == "__main__":
    """
    Main function to initialize, test, and inspect the ReasoningTransformer.
    """
    print("--- Initializing Model and Dummy Components ---")

    # --- Model Hyperparameters ---
    D_POLICY = 1929  # Vocabulary size, e.g. for all possible moves
    D_MODEL = 1024     # Main hidden dimension
    N_LAYERS = 4       # Number of Transformer layers
    N_HEADS = 16       # Number of attention heads
    D_FF = 4096        # Feed-forward dimension (4 * D_MODEL)
    DROPOUT = 0
    
    # --- Data Shape Hyperparameters ---
    SEQ_LEN = 64       # Fixed sequence length (N)
    BATCH_SIZE = 2     # Number of games in a batch (B)
    CHESS_TENSOR_SHAPE = (8,8,19) # Shape of a single board state (C)

    # --- Dummy User-Provided Modules (for testing) ---
    # In your actual project, you would use your real, pre-trained modules.
    class DummyFenEncoder(nn.Module):
        """Mocks the FEN encoder, returning a tensor of the correct shape."""
        def __init__(self, d_model):
            super().__init__()
            # A simple linear layer is enough to simulate the embedding process.
            # We'll use a dummy input size of 10.
            self.linear = nn.Linear(10, d_model)
        def forward(self, fens: List[str]):
            # Create a random tensor to simulate the output of a real FEN processor.
            dummy_input = torch.randn(len(fens), 10,device = "cuda")
            return self.linear(dummy_input)


    model = ReasoningTransformer(
    d_policy=D_POLICY,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    dropout_p=DROPOUT,
    N=SEQ_LEN
    ).to("cuda")

    # --- Attach the User's External Modules ---
    # The main model class assumes these modules are attached to it.
    #model.fen_encoder = DummyFenEncoder(D_MODEL).to("cuda")

    # --- Calculate and Print the Number of Parameters ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel instantiated successfully.")
    print(f"Total trainable parameters: {total_params:,} (~{total_params / 1_000_000:.2f}M)")

    # --- Generate Random Input for Inference ---
    print(f"\n--- Generating Random Input for Inference (Batch Size={BATCH_SIZE}) ---")
    
    # 1. Precomputed states (B, N, *C)
    dummy_states = torch.randn(BATCH_SIZE, SEQ_LEN, *CHESS_TENSOR_SHAPE).to("cuda")
    
    # 2. Target tokens (B, N)
    dummy_tokens = torch.randint(0, D_POLICY, (BATCH_SIZE, SEQ_LEN)).to("cuda")
    
    # 3. List of FEN strings (length B)
    dummy_fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * BATCH_SIZE
    
    # Package the inputs into the tuple format expected by the model
    inp = (dummy_states, dummy_tokens, dummy_fens)
    
    print(f"Input `states` shape: {dummy_states.shape}")
    print(f"Input `tokens` shape: {dummy_tokens.shape}")
    print(f"Input `fens` length: {len(dummy_fens)}")

    # --- Perform a Forward Pass (Inference) ---
    print("\n--- Performing a Forward Pass ---")
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for inference
        logits = model(inp)
    
    print(f"Output `logits` shape: {logits.shape}")
    
    # --- Verify Output Shape ---
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, D_POLICY)
    print("\nForward pass successful. The model works as expected.")
