import torch
import sys
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
sys.path.append(os.getcwd())
try:
    from .attn import RelativeMultiHeadAttention2 # User changed from .attn
except:
    from attn import RelativeMultiHeadAttention2 # User changed from .attn
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

        #self.final_decoder = PolicyRefinerDecoder(1929,64*4)

    def forward(self,inp,compute_loss = False):
        x = inp[0]
        b,seq_len,_,_,emb = x.size()
        x = x.view(b*seq_len,64,emb)
    
        x = self.linear1(x)
        #add gelu
        x = torch.nn.GELU()(x)

        x = self.layernorm1(x)

        #add ma gating 
        x = self.ma_gating(x)
        pos_enc = self.positional(x)
        for i in range(self.num_layers):
            x = self.layers[i](x,pos_enc)
        
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
            loss = F.cross_entropy(policy.view(-1,policy.size(-1)), targets.view(-1),ignore_index=1928)
            return policy, loss, targets
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
            indexes = np.argsort(-probs[0].cpu().detach().numpy())
            for index in indexes[:10]:
                    print(policy_index[index],probs[0][index].item())
            #sample a move according to the probabilities
            #print(probs.shape)
            sampled = torch.multinomial(probs, num_samples=1)
        move = policy_index[sampled.item()]

        return move


if __name__ == "__main__":
    model = BT4().to("cuda")
    # Input tensor shape: (batch_size, sequence_length, board_dim1, board_dim2, embedding_dim)
    # For the model, this becomes (batch_size * sequence_length, board_dim1 * board_dim2, embedding_dim)
    # (4, 64, 8, 8, 19) -> (256, 64, 19)
    random_tensor = torch.randn(4, 64, 8, 8, 19).to("cuda")
    output = model(random_tensor)
    print("Model output shape:", output.shape)
    print("Model output (first element of batch):", output[0])

