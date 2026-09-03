from __future__ import annotations

from typing import Callable

from torch import nn

from src.models.frame import FrameModel
from src.models.backbone import BACKBONES
from src.models.temporal import TemporalModel
from src.models.mae_face import build_mae_face
from src.utils import build_transform, to_chw_float

__all__ = ["FrameModel", "TemporalModel", "BACKBONES", "build_model", "input_transform"]


def build_model(config, n_classes: int, *, pretrained: bool = True,
                freeze_backbone: bool = False, dropout: float = 0.0) -> nn.Module:
    """Build the model named by config.model -> (B, n_classes) logits.
      resnet50 | densenet121         -> FrameModel  (unit_type: frame)
      resnet50_tcn | densenet121_tcn -> TemporalModel + causal TCN (unit_type: window)
    """
    if config.mode == "frame":
        return FrameModel(n_classes=n_classes, backbone=config.model, pretrained=pretrained,
                          freeze_backbone=freeze_backbone, dropout=dropout)
    if config.model.endswith("_tcn"):
        backbone = config.model[: -len("_tcn")]
        return TemporalModel(n_classes=n_classes, backbone=backbone, pretrained=pretrained,
                             freeze_backbone=freeze_backbone)
    raise NotImplementedError(f"no builder yet for model={config.model!r}")


def input_transform(config, *, train: bool = False) -> Callable:
    """The dataset transform for the selected model. Frame CNNs and the CNN+TCN
    temporal model share the same per-frame ImageNet-normalized transform (the TCN
    runs the CNN on each frame). V-JEPA will need its own."""
    if config.mode == "frame" or config.model.endswith("_tcn"):
        return build_transform(size=config.img_size, train=train, normalize=True)
    raise NotImplementedError(f"no transform yet for model={config.model!r}")