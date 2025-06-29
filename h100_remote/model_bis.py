"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from utils.vocab import policy_index
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.fen_encoder import FenEncoder
from utils.parse import encode_fens
import chess
start_think_index = policy_index.index("<thinking>")
end_think_index = policy_index.index("</thinking>")
end_variation_index = policy_index.index("end_variation")
end_index = policy_index.index("end")

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
        
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.FenEncoder = FenEncoder(config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Initialize fixed positional encodings
        position = torch.arange(config.block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
        pe = torch.zeros(config.block_size, config.n_embd)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)
        # Apply scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, batch, compute_loss=False):
        position = self.FenEncoder(batch[0])
        targets_pos = torch.full((position.shape[0], 64), 1928, device=batch[1].device, dtype=torch.int64)
        idx = batch[1]
        targets = torch.cat((targets_pos, idx), dim=1)
        targets = torch.roll(targets, shifts=-1, dims=1)
        device = idx.device
        b, t = idx.size()
        t += 64
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward through the model
        tok_emb = self.transformer.wte(idx)
        tok_emb = torch.cat((position, tok_emb), dim=1)
        pos_emb = self.pe[pos]
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if compute_loss:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=1928)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, targets

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.pe = self.pe[:block_size]
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # The rest of the methods (from_pretrained, configure_optimizers, estimate_mfu, generate, generate_sequence) remain unchanged
    # except for replacing any references to wpe with pe in generate_sequence

    @torch.no_grad()
    def generate_sequence(self, board, T=1):
        position = self.FenEncoder(board)
        sequences = []
        probs = []
        tok_emb = position
        device = position.device
        b, t, _ = position.shape  # t = 64
        
        # Append think token
        idx_next = torch.full((tok_emb.shape[0], 1), start_think_index, device=device)
        sequences.append(idx_next)
        tok_emb = torch.cat((tok_emb, self.transformer.wte(idx_next)), dim=1)
        
        end_think = torch.full((tok_emb.shape[0], 1), end_think_index, device=device)
        end = torch.full((tok_emb.shape[0], 1), end_index, device=device)
        num_zeros = torch.zeros((tok_emb.shape[0], 1), device=device)
        t += 1
        
        for _ in range(self.config.block_size - 64 - 1):
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.pe[pos]  # Using fixed positional encoding
            x = self.transformer.drop(tok_emb + pos_emb)
            
            for block in self.transformer.h:
                x = block(x)
            
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.view(b, -1)
            
            if T == 0:
                #print("using argmax")
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                prob = F.softmax(logits / T, dim=-1)
                idx_next = torch.multinomial(prob, num_samples=1)
                probs.append(prob.unsqueeze(1))
            
            # If num_zeros == 5, change idx to stop thinking
            idx_next = torch.where(num_zeros == 5, end_think, idx_next)
            num_zeros = torch.where(num_zeros == 5, 6, num_zeros)  # Change 5s to 6 after updating idx
            
            if len(sequences) > 1:  # If before last is </think>, place end token.
                to_end = (sequences[-2] == end_think_index)
                idx_next = torch.where(to_end, end, idx_next)
            
            num_zeros = num_zeros + (idx_next == end_variation_index).float()
            sequences.append(idx_next)
            t += 1
            tok_emb = torch.cat((tok_emb, self.transformer.wte(idx_next)), dim=1)
        
        sequences = torch.cat(sequences, dim=1)
        if T != 0:
            probs = torch.cat(probs, dim=1)
        else:
            probs = None
        
        return sequences, probs

    @torch.no_grad()
    def generate_sequence_raw(self, board, T=1, return_logits = False):
        position = self.FenEncoder(board)
        sequences = []
        probs = []
        tok_emb = position
        device = position.device
        b, t, _ = position.shape  # t = 64
        
        # Append think token
        idx_next = torch.full((tok_emb.shape[0], 1), start_think_index, device=device)
        sequences.append(idx_next)
        tok_emb = torch.cat((tok_emb, self.transformer.wte(idx_next)), dim=1)
        t += 1
        
        for _ in range(self.config.block_size - 64 - 1):
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.pe[pos]  # Using fixed positional encoding
            x = self.transformer.drop(tok_emb + pos_emb)
            
            for block in self.transformer.h:
                x = block(x)
            
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.view(b, -1)
            
            if T == 0:
                #print("using argmax")
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                prob = F.softmax(logits / T, dim=-1)
                idx_next = torch.multinomial(prob, num_samples=1)
                if return_logits:
                    probs.append(logits.unsqueeze(1))
                else:
                    probs.append(prob.unsqueeze(1))
            
            sequences.append(idx_next)
            t += 1
            tok_emb = torch.cat((tok_emb, self.transformer.wte(idx_next)), dim=1)
        
        sequences = torch.cat(sequences, dim=1)
        if T != 0:
            probs = torch.cat(probs, dim=1)
        else:
            probs = None
        
        return sequences, probs
    
    def get_move_from_fen(self, fen, T = 1,device = "cuda",force_legal = True,return_probs = False):
        fen_array = [fen]
        encoded_board = encode_fens(fen_array).to(device)
        id_model, logits = self.generate_sequence_raw(encoded_board, T=T, return_logits=True)
        list_moves_model = []
        for i,move in enumerate(id_model[0]):
            list_moves_model.append(policy_index[move])
            #print(policy_index[move])
            if policy_index[move]== "end":
                break
        logits = logits[0][i-2]# since we are on the probabilities.
        #calculate legal_move_mask
        board = chess.Board(fen)
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
        #softmax
        probs = F.softmax(logits/T, dim=-1)
        #sample a move according to the probabilities
        #print(probs.shape)
        sampled = torch.multinomial(probs, num_samples=1)
        #get the move
        move = policy_index[sampled.item()]
        if return_probs:
            return move, probs
        else:
            return move
    

    def get_move_from_fen_no_thinking(self, fen, T = 1,device = "cuda",force_legal = True,return_probs = False):
        fen_array = [fen]
        encoded_board = encode_fens(fen_array).to(device)
        encoded_board = self.FenEncoder(encoded_board)
        t= 64
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.pe[pos]
        #print(encoded_board.shape,pos_emb.shape)
        x = self.transformer.drop(encoded_board + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :])[0]
        #print(logits.shape)
        #calculate legal_move_mask
        board = chess.Board(fen)
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
        #softmax
        if T == 0:
            #print("using argmax")
            sampled = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits/T, dim=-1)
            #sample a move according to the probabilities
            #print(probs.shape)
            sampled = torch.multinomial(probs, num_samples=1)
        #get the move
        move = policy_index[sampled.item()]
        if return_probs:
            return move, probs
        else:
            return move