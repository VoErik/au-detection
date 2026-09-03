from __future__ import annotations

"""
Frame-based AU baseline: an ImageNet-pretrained backbone + a linear K-way head,
returning (B, K) logits (one independent sigmoid per AU -- multi-label, so all-zero
"neutral" frames are just all-negative targets).

Backbones: resnet50, densenet121, vit_b_16. These want ImageNet-NORMALIZED input;
build the dataset transform via models.input_transform (or utils.build_transform
with normalize=True), not the [0,1] default.
"""

import torch
from torch import nn

from src.models.backbone import _build_backbone


class FrameModel(nn.Module):
    """Backbone -> linear K-way head -> (B, K) logits.

    freeze_backbone=True is a linear probe: backbone frozen and kept in eval() even
    under .train() so its BatchNorm stats / dropout don't drift. False is full
    fine-tuning (the stronger baseline)."""

    def __init__(self, n_classes: int, backbone: str = "densenet121", *,
                 pretrained: bool = True, freeze_backbone: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.backbone, feat = _build_backbone(backbone, pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.head = (nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, n_classes))
                     if dropout > 0 else nn.Linear(feat, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,3,H,W) -> (B,K)
        if self.freeze_backbone:
            with torch.no_grad():
                f = self.backbone(x)
        else:
            f = self.backbone(x)
        return self.head(f)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self