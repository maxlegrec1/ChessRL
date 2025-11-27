
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vocab import policy_index, policy_to_idx
from .fen_encoder import fen_to_tensor
import bulletchess

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = F.relu(x)
        return x

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv2d(19, 64, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(64)
        
        # 4 Residual Blocks
        self.blocks = nn.ModuleList([ResBlock(64) for _ in range(4)])
        
        # Policy Head
        # Reduce channels to 16, then flatten to 1024
        self.policy_conv = nn.Conv2d(64, 16, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.policy_fc2 = nn.Linear(256, 1929)
        
        # Value Head
        self.value_conv = nn.Conv2d(64, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 3) # WDL

    def forward(self, inp, *, compute_loss: bool = False, return_all: bool = False, step = None):
        # inp is a list/tuple: [x, targets, ...] OR just x
        if isinstance(inp, (list, tuple)):
            x = inp[0]
        else:
            x = inp
        
        # Handle sequence dimension if present
        if x.dim() == 5:
            b, seq, h, w, c = x.shape
            x = x.view(b * seq, h, w, c)
            has_seq = True
        else:
            has_seq = False
            
        # Permute to (B, C, H, W) for Conv2d
        x = x.permute(0, 3, 1, 2)
        
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        for block in self.blocks:
            x = block(x)
            
        # Policy Head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.flatten(1)
        p = self.policy_fc1(p)
        p = F.relu(p)
        policy_logits = self.policy_fc2(p)
        
        # Value Head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.flatten(1)
        v = self.value_fc1(v)
        v = F.relu(v)
        value_logits = self.value_fc2(v)
        
        # For compatibility with trainer which expects value_q_logits as well
        # We can just return value_logits twice or zeros
        value_q_logits = value_logits
        
        if has_seq:
            b = inp[0].shape[0]
            seq = inp[0].shape[1]
            policy_logits = policy_logits.view(b, seq, -1)
            value_logits = value_logits.view(b, seq, -1)
            value_q_logits = value_q_logits.view(b, seq, -1)

        if return_all:
            return policy_logits, value_logits, value_q_logits
            
        return policy_logits

    def get_move_from_fen_no_thinking(self, fen: str, T: float = 0.0, device: str = None, return_probs: bool = False):
        if device is None:
            device = next(self.parameters()).device
        
        # Encode FEN
        tensor = fen_to_tensor(fen)
        x = torch.from_numpy(tensor).unsqueeze(0).to(device) # [1, 8, 8, 19]
        
        self.eval()
        with torch.no_grad():
            logits = self(x) # [1, 1929]
            
        # Mask illegal moves
        board = bulletchess.Board.from_fen(fen)
        legal_mask = torch.full((1, 1929), -float('inf'), device=device)
        
        legal_moves_found = False
        for move in board.legal_moves():
            uci = move.uci()
            if uci.endswith('n'): # Promote to knight is just 'e7e8n' -> 'e7e8' in our vocab usually? 
                # Wait, vocab policy_to_idx handles 'e7e8n' if it's in there.
                # Let's check how policy_to_idx is defined. 
                # Assuming standard behavior from previous model:
                pass
            
            # In previous model code:
            # if u[-1] == 'n': u = u[:-1]
            # This implies knight promotions are mapped to the base move or handled differently?
            # Actually, standard UCI for promotion is e7e8q. 
            # If our vocab doesn't have specific promotion moves, we might need to strip.
            # But let's stick to what the other model did.
            if uci.endswith('n'):
                uci = uci[:-1]
                
            if uci in policy_to_idx:
                idx = policy_to_idx[uci]
                legal_mask[0, idx] = 0.0
                legal_moves_found = True
                
        if not legal_moves_found:
            # Should not happen for valid FENs unless game over
            return None if not return_probs else (None, None)
            
        logits = logits + legal_mask
        
        # Softmax
        probs = F.softmax(logits / (T if T > 0 else 1.0), dim=-1)
        
        if T == 0:
            # Greedy
            move_idx = torch.argmax(probs, dim=-1).item()
        else:
            # Sample
            move_idx = torch.multinomial(probs, 1).item()
            
        move_uci = policy_index[move_idx]
        
        if return_probs:
            return move_uci, probs
            
        return move_uci

    def get_position_value(self, fen: str, device: str = None):
        if device is None:
            device = next(self.parameters()).device
            
        tensor = fen_to_tensor(fen)
        x = torch.from_numpy(tensor).unsqueeze(0).to(device)
        
        self.eval()
        with torch.no_grad():
            _, value_logits, _ = self(x, return_all=True)
            value_probs = torch.softmax(value_logits, dim=-1)
            
        # Return WDL
        return value_probs[0].tolist()
