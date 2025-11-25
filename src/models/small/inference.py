import torch
import sys
import os
import bulletchess
import torch.nn.functional as F
import torch.nn as nn
import concurrent.futures as _futures

from .attn import RelativeMultiHeadAttention2
from .vocab import policy_index, policy_to_idx
from .fen_encoder import fen_to_tensor, batch_fen_to_tensor
from typing import List, Tuple, Union

class MaGating(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(64, d_model))
        self.b = torch.nn.Parameter(torch.ones(64, d_model))
        self.register_buffer("_exp_a", torch.exp(self.a.detach()), persistent=False)

    def forward(self,x):
        return x*self._exp_a + self.b

    def rebuild_exp_a(self):
        with torch.no_grad():
            self._exp_a.copy_(torch.exp(self.a.detach()))

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

class Inference(torch.nn.Module):
    def __init__(self,num_layers = 4, d_model=256, d_ff=512, num_heads = 4):
        super().__init__()
        self.is_thinking_model = False
        self.d_model = d_model

        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList([EncoderLayer(d_model,d_ff,num_heads) for _ in range(num_layers)])
        self.linear1 = torch.nn.Linear(19,d_model)

        self.layernorm1 = torch.nn.LayerNorm(d_model)

        self.policy_tokens_lin = torch.nn.Linear(d_model,d_model)

        self.queries_pol = torch.nn.Linear(d_model,d_model)

        self.keys_pol = torch.nn.Linear(d_model,d_model)

        self.positional = AbsolutePositionalEncoder(d_model)

        self.ma_gating = MaGating(d_model) # Pass d_model
        self.ma_gating.rebuild_exp_a()

        self.policy_head = torch.nn.Linear(64*64,1929,bias=False)

        self.value_head = ValueHead(d_model)
        self.value_head_q = ValueHeadQ(d_model)

    def forward(self,x):
        x = x.view(-1,64,19)
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

        value = self.value_head_q(x)
        value = torch.softmax(value, dim=-1)

        policy_tokens = self.policy_tokens_lin(x)
        policy_tokens = torch.nn.GELU()(policy_tokens)
        policy_tokens = policy_tokens + pos_enc
        queries = self.queries_pol(policy_tokens)

        keys = self.keys_pol(policy_tokens)

        matmul_qk = torch.matmul(queries,torch.transpose(keys,-2,-1))

        dk = torch.sqrt(torch.tensor(self.d_model, dtype=matmul_qk.dtype, device=matmul_qk.device))

        policy_attn_logits = matmul_qk / dk
        policy_attn_logits = policy_attn_logits.view(-1,64*64)

        policy = self.policy_head(policy_attn_logits)#shape (-1,1929)

        return policy, value

    def load_state_dict(self, state_dict, strict: bool = True):
        load_return = super().load_state_dict(state_dict, strict)
        self.ma_gating.rebuild_exp_a()
        return load_return

    @staticmethod
    def _gumbel_sample(logits: torch.Tensor, T: float, topk: int = 0) -> torch.Tensor:
        """
        Gumbel-max sampling. If topk>0, restrict to top-k logits first.
        Returns selected indices (B,).
        """
        if topk and topk > 0 and topk < logits.shape[-1]:
            topk_vals, topk_idx = torch.topk(logits, k=topk, dim=-1)
            g = -torch.log(-torch.log(torch.rand_like(topk_vals)))
            sel_local = torch.argmax(topk_vals / T + g, dim=-1)
            return topk_idx.gather(1, sel_local.unsqueeze(1)).squeeze(1)
        g = -torch.log(-torch.log(torch.rand_like(logits)))
        return torch.argmax(logits / T + g, dim=-1)

    def get_move_from_fen_no_thinking(self, fen, T = 1,device = "cuda",force_legal = True,return_probs = False, use_fp16: bool = False):
        board = bulletchess.Board.from_fen(fen)
        x = torch.from_numpy(fen_to_tensor(fen)).to(device).to(torch.float32)
        x = x.view(1,8,8,19)

        if use_fp16 and torch.device(device).type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits,_ = self(x.half())
        else:
            logits,_ = self(x)
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
            if return_probs:
                probs = F.softmax(logits/T, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                return sampled, probs
            sampled = self._gumbel_sample(logits, T=T, topk=128)
        move = policy_index[sampled.item()]

        return move

    def get_position_value(self, fen, device="cuda"):
        """
        Get the value evaluation for a given FEN position.
        Returns the value vector [black_win_prob, draw_prob, white_win_prob]
        """
        x = torch.from_numpy(fen_to_tensor(fen)).to(device).to(torch.float32)
        x = x.view(1, 8, 8, 19)
        
        # Forward pass through the model to get value
        with torch.no_grad():
            # We need to run through the model layers to get to value_head
            b = x.shape[0]
            emb = x.shape[-1]  # Last dimension (should be 19)
            x_processed = x.view(b, 64, emb)
            x_processed = self.linear1(x_processed)
            x_processed = torch.nn.GELU()(x_processed)
            x_processed = self.layernorm1(x_processed)
            x_processed = self.ma_gating(x_processed)
            
            pos_enc = self.positional(x_processed)
            for i in range(self.num_layers):
                x_processed = self.layers[i](x_processed, pos_enc)
            
            value_logits = self.value_head_q(x_processed)
            value_logits = value_logits.view(b, 3)
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
        
        # Forward pass through the model to get values
        with torch.no_grad():
            b = batch_x.shape[0]
            emb = batch_x.shape[-1]  # Last dimension (should be 19)
            x_processed = batch_x.view(b, 64, emb)
            x_processed = self.linear1(x_processed)
            x_processed = torch.nn.GELU()(x_processed)
            x_processed = self.layernorm1(x_processed)
            x_processed = self.ma_gating(x_processed)
            
            pos_enc = self.positional(x_processed)
            for i in range(self.num_layers):
                x_processed = self.layers[i](x_processed, pos_enc)
            
            value_logits = self.value_head_q(x_processed)
            value_logits = value_logits.view(b, 3)
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

    def batch_get_moves_from_fens(self, fens: List[str], T: float = 0.0, device: str = None, force_legal: bool = True, return_probs: bool = False, use_fp16: bool = False, boards: List["bulletchess.Board"] = None, topk_prefilter: int = 0) -> Union[List[str], Tuple[List[str], torch.Tensor]]:
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        if not fens:
            if return_probs:
                return [], torch.empty(0, 1929, device=device)
            return []

        # Encode all FENs to a batch tensor: (B, 8, 8, 19)
        batch_x = batch_fen_to_tensor(fens, device=device.type if hasattr(device, "type") else str(device), dtype=torch.float16 if (use_fp16 and torch.device(device).type == 'cuda') else torch.float32)
        # Forward pass -> logits (B, 1929)
        self.eval()
        with torch.inference_mode():
            if use_fp16 and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits,_ = self(batch_x.half())
            else:
                logits,_ = self(batch_x)
        B = logits.shape[0]
        logits = logits.view(B, 1929)

        # Build legal move masks per position using Bullet Chess
        if force_legal:
            def board_from_fen(f: str) -> "bulletchess.Board":
                return bulletchess.Board.from_fen(f)
            boards_local = boards if boards is not None else [board_from_fen(f) for f in fens]

            # Optionally prefilter to top-k indices for legality checks to reduce CPU work
            if topk_prefilter and topk_prefilter > 0 and topk_prefilter < logits.shape[-1]:
                topk_vals, topk_idx = torch.topk(logits, k=topk_prefilter, dim=-1)
                legal_mask = torch.full((B, 1929), -999.0, device=device, dtype=logits.dtype)

                def legal_set_for_row(args):
                    row, board = args
                    legal_set = set()
                    for mv in board.legal_moves():
                        u = mv.uci()
                        if u[-1] == 'n':
                            u = u[:-1]
                        idx = policy_to_idx.get(u)
                        if idx is not None:
                            legal_set.add(idx)
                    return row, legal_set

                with _futures.ThreadPoolExecutor() as ex:
                    for row, legal in ex.map(legal_set_for_row, [(i, b) for i, b in enumerate(boards_local)]):
                        # Only zero out allowed top-k logits if in legal set
                        row_topk = topk_idx[row].tolist()
                        for idx in row_topk:
                            if idx in legal:
                                legal_mask[row, idx] = 0.0
                logits = logits + legal_mask
            else:
                legal_mask = torch.full((B, 1929), -999.0, device=device, dtype=logits.dtype)

                def write_legal_row(args):
                    row, board = args
                    for mv in board.legal_moves():
                        u = mv.uci()
                        if u[-1] == 'n':
                            u = u[:-1]
                        idx = policy_to_idx.get(u)
                        if idx is not None:
                            legal_mask[row, idx] = 0.0

                with _futures.ThreadPoolExecutor() as ex:
                    list(ex.map(write_legal_row, [(i, b) for i, b in enumerate(boards_local)]))
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
            if return_probs:
                scaled = logits / T
                probs_tensor = F.softmax(scaled, dim=1)
                sampled = torch.multinomial(probs_tensor, num_samples=1).squeeze(1)
                for idx in sampled.tolist():
                    moves.append(policy_index[idx])
            else:
                sampled = self._gumbel_sample(logits, T=T, topk=128)
                for idx in sampled.tolist():
                    moves.append(policy_index[idx])

        if return_probs:
            return moves, probs_tensor
        return moves

    def batched_move_selection(
        self,
        fens: List[str] = None,
        boards: List["bulletchess.Board"] = None,
        batch_x: torch.Tensor = None,
        T: float = 0.0,
        device: str = None,
        force_legal: bool = True,
        return_probs: bool = False,
        use_fp16: bool = False,
        topk_prefilter: int = 128,
        return_indices: bool = False,
    ) -> Union[List[str], Tuple[List[str], torch.Tensor], List[int], Tuple[List[int], torch.Tensor]]:
        """
        Fused path: build/update inputs, run model, apply legality (optional), and sample.
        Returns move UCIs by default; set return_indices=True to return policy indices instead.
        """
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        if batch_x is None:
            if boards is not None:
                from .fen_encoder import boards_to_tensor
                batch_x = boards_to_tensor(boards, device=device.type, dtype=torch.float16 if (use_fp16 and device.type == 'cuda') else torch.float32)
            elif fens is not None:
                batch_x = batch_fen_to_tensor(fens, device=device.type, dtype=torch.float16 if (use_fp16 and device.type == 'cuda') else torch.float32)
            else:
                return [] if not return_probs else ([], torch.empty(0, 1929, device=device))

        self.eval()
        with torch.inference_mode():
            if use_fp16 and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = self(batch_x.half())
            else:
                logits = self(batch_x)

        B = logits.shape[0]
        logits = logits.view(B, 1929)

        if force_legal:
            def board_from_fen(f: str) -> "bulletchess.Board":
                return bulletchess.Board.from_fen(f)
            boards_local = boards if boards is not None else [board_from_fen(f) for f in (fens or [])]
            legal_mask = torch.full((B, 1929), -999.0, device=device, dtype=logits.dtype)

            def write_legal_row(args):
                row, board = args
                for mv in board.legal_moves():
                    u = mv.uci()
                    if u[-1] == 'n':
                        u = u[:-1]
                    idx = policy_to_idx.get(u)
                    if idx is not None:
                        legal_mask[row, idx] = 0.0

            with _futures.ThreadPoolExecutor() as ex:
                list(ex.map(write_legal_row, [(i, b) for i, b in enumerate(boards_local)]))
            logits = logits + legal_mask

        if T == 0.0:
            best_idx = torch.argmax(logits, dim=1)
            if return_probs:
                probs_tensor = torch.zeros_like(logits)
                probs_tensor[torch.arange(B), best_idx] = 1.0
            selected = best_idx
        else:
            if return_probs:
                probs_tensor = F.softmax(logits / T, dim=1)
                selected = torch.multinomial(probs_tensor, num_samples=1).squeeze(1)
            else:
                selected = self._gumbel_sample(logits, T=T, topk=topk_prefilter)

        if return_indices:
            return (selected.tolist(), probs_tensor) if return_probs else selected.tolist()

        moves: List[str] = [policy_index[i] for i in selected.tolist()]
        return (moves, probs_tensor) if return_probs else moves

class ValueHead(torch.nn.Module):
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
