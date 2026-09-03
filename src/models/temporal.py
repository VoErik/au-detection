from __future__ import annotations

"""
Temporal head: run the shared per-frame CNN backbone over an L-frame window to get
(B, L, feat), mix across time with a CAUSAL TCN (dilated 1D convs, left-padded so
timestep t sees only t and earlier), and read out the LAST timestep -> (B, K)
logits. Causal + short receptive field matches AU onset/offset dynamics, and the
last-timestep readout matches the "predict frame n from n-L+1..n" convention.

Input is (B, L, 3, H, W) -- AUDataset with unit_type="window". Output is (B, K),
the same contract as FrameModel, so it slots into the training harness unchanged.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbone import _build_backbone


class CausalConv1d(nn.Module):
    """Conv1d over time, left-padded by (kernel-1)*dilation so it's causal."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, C, L)
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """Two causal convs + residual, at a fixed dilation."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, kernel, dilation)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = CausalConv1d(c_out, c_out, kernel, dilation)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(F.relu(self.bn1(self.conv1(x))))
        y = self.drop(F.relu(self.bn2(self.conv2(y))))
        return F.relu(y + self.down(x))


class TemporalModel(nn.Module):
    """CNN backbone (per frame) -> causal TCN (over time) -> head on last timestep.

    tcn_layers dilations double (1,2,4,...), so the last timestep's receptive field
    grows to cover the window. freeze_backbone trains only the TCN+head on frozen
    per-frame features (cheap; the backbone stays in eval so its BN doesn't drift)."""

    def __init__(self, n_classes: int, backbone: str = "densenet121", *,
                 pretrained: bool = True, freeze_backbone: bool = False,
                 tcn_channels: int = 256, tcn_layers: int = 4, kernel: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.backbone, feat = _build_backbone(backbone, pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        blocks, c = [], feat
        for i in range(tcn_layers):
            blocks.append(TCNBlock(c, tcn_channels, kernel, dilation=2 ** i, dropout=dropout))
            c = tcn_channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(tcn_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (B, L, 3, H, W) -> (B, K)
        B, L = x.shape[:2]
        frames = x.flatten(0, 1)                              # (B*L, 3, H, W)
        if self.freeze_backbone:
            with torch.no_grad():
                f = self.backbone(frames)
        else:
            f = self.backbone(frames)
        f = f.view(B, L, -1).transpose(1, 2)                  # (B, feat, L)
        f = self.tcn(f)                                       # (B, C, L)
        return self.head(f[:, :, -1])                         # last timestep

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self