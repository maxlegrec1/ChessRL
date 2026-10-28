import torch.nn as nn
import torch.nn.functional as F

from .fen_encoder import INPUT_PLANES


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
    """AlphaZero-style ResNet with five heads. Default trunk: 4 blocks x 64 channels.

    `blocks` / `channels` size the trunk; heads keep their fixed reduction widths
    so checkpoints of any trunk size share head layouts (see src/models/growth.py)."""

    def __init__(self, blocks: int = 4, channels: int = 64):
        super().__init__()
        self.num_blocks = blocks
        self.channels = channels
        self.input_conv = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(blocks)])

        # Policy Head
        # Reduce channels to 16, then flatten to 1024
        self.policy_conv = nn.Conv2d(channels, 16, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.policy_fc2 = nn.Linear(256, 1929)

        # Value Head
        self.value_conv = nn.Conv2d(channels, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 3)  # WDL

        # Value Head Q
        self.value_conv_q = nn.Conv2d(channels, 8, 1, bias=False)
        self.value_bn_q = nn.BatchNorm2d(8)
        self.value_fc1_q = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2_q = nn.Linear(128, 3)

        # Moves-Left Heads. Both predict a raw scalar interpretable as
        # plies_remaining / 25 (linear; no activation baked in). The C++ MCTS
        # multiplies by 25 for the depth-incremented backup, and applies tanh
        # at the tiebreaker call site to bound the modifier.
        #   mlh   ← true target: (game_total_plies - ply_index) / 25.
        #   mlh_q ← search-backed target: MCTS-backed-up plies / 25 at root.
        # Mirrors the value/value_q split: one unbiased anchor, one search-aware
        # estimate. ExportWrapper picks which one feeds the engine (mlh under
        # search_heads=outcome, mlh_q under search_heads=search).
        self.mlh_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.mlh_bn = nn.BatchNorm2d(4)
        self.mlh_fc1 = nn.Linear(4 * 8 * 8, 64)
        self.mlh_fc2 = nn.Linear(64, 1)

        self.mlh_q_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.mlh_q_bn = nn.BatchNorm2d(4)
        self.mlh_q_fc1 = nn.Linear(4 * 8 * 8, 64)
        self.mlh_q_fc2 = nn.Linear(64, 1)

    def spec(self) -> dict:
        return {"arch": "resnet", "blocks": self.num_blocks, "channels": self.channels}

    def forward(self, x, *, return_all: bool = False):
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

        # Value Head Q
        v_q = self.value_conv_q(x)
        v_q = self.value_bn_q(v_q)
        v_q = F.relu(v_q)
        v_q = v_q.flatten(1)
        v_q = self.value_fc1_q(v_q)
        v_q = F.relu(v_q)
        value_q_logits = self.value_fc2_q(v_q)

        # Moves-Left heads (linear). Output is interpretable as plies/25.
        m = self.mlh_conv(x)
        m = self.mlh_bn(m)
        m = F.relu(m)
        m = m.flatten(1)
        m = self.mlh_fc1(m)
        m = F.relu(m)
        mlh = self.mlh_fc2(m).squeeze(-1)

        m_q = self.mlh_q_conv(x)
        m_q = self.mlh_q_bn(m_q)
        m_q = F.relu(m_q)
        m_q = m_q.flatten(1)
        m_q = self.mlh_q_fc1(m_q)
        m_q = F.relu(m_q)
        mlh_q = self.mlh_q_fc2(m_q).squeeze(-1)

        if return_all:
            return policy_logits, value_logits, value_q_logits, mlh, mlh_q

        return policy_logits
