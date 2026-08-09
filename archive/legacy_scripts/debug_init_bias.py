import torch
from src.models.small.model import BT4
import numpy as np

def main():
    print("Initializing new BT4 model...")
    model = BT4(num_layers=4, d_model=256, d_ff=512, num_heads=4).to("cpu")
    model.eval()
    
    print("Generating random inputs...")
    # Input shape: (batch, seq_len, 8, 8, 19)
    # Model expects (batch, 1, 8, 8, 19) or similar depending on usage
    # In get_batch_position_values: (batch, 1, 8, 8, 19)
    
    batch_size = 1000
    # Random binary features (simulating FENs roughly)
    x = torch.randn(batch_size, 1, 8, 8, 19)
    
    print("Running inference...")
    with torch.no_grad():
        # Forward pass
        # model(x) returns policy
        # We need value head output.
        # Let's use get_batch_position_values logic but with raw tensor
        
        # From model.py:
        b = x.shape[0]
        seq_len = x.shape[1]
        emb = x.shape[-1]
        x_processed = x.view(b * seq_len, 64, emb)
        x_processed = model.linear1(x_processed)
        x_processed = torch.nn.GELU()(x_processed)
        x_processed = model.layernorm1(x_processed)
        x_processed = model.ma_gating(x_processed)
        
        pos_enc = model.positional(x_processed)
        for i in range(model.num_layers):
            x_processed = model.layers[i](x_processed, pos_enc)
        
        # Check both value heads
        v1 = model.value_head(x_processed) # (B, 3)
        v2 = model.value_head_q(x_processed) # (B, 3)
        
        # Softmax
        probs1 = torch.softmax(v1, dim=-1)
        probs2 = torch.softmax(v2, dim=-1)
        
    # Analyze argmax
    argmax1 = torch.argmax(probs1, dim=-1).numpy()
    argmax2 = torch.argmax(probs2, dim=-1).numpy()
    
    counts1 = np.bincount(argmax1, minlength=3)
    counts2 = np.bincount(argmax2, minlength=3)
    
    print("\nValue Head 1 (Standard):")
    print(f"Win (0): {counts1[0]} ({counts1[0]/batch_size:.1%})")
    print(f"Draw (1): {counts1[1]} ({counts1[1]/batch_size:.1%})")
    print(f"Loss (2): {counts1[2]} ({counts1[2]/batch_size:.1%})")
    
    print("\nValue Head 2 (Q):")
    print(f"Win (0): {counts2[0]} ({counts2[0]/batch_size:.1%})")
    print(f"Draw (1): {counts2[1]} ({counts2[1]/batch_size:.1%})")
    print(f"Loss (2): {counts2[2]} ({counts2[2]/batch_size:.1%})")
    
    # Check average probabilities
    avg1 = probs1.mean(dim=0).numpy()
    avg2 = probs2.mean(dim=0).numpy()
    
    print("\nAverage Probabilities (Head 1):", avg1)
    print("Average Probabilities (Head 2):", avg2)

if __name__ == "__main__":
    main()
