from __future__ import annotations

import random
from typing import Callable, Sequence

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_chw_float(img_rgb: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float32 in [0,1]. The zero-config transform (no resize,
    no normalize); crops are already the right size, so this is the default."""
    return torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1).float().div_(255.0)


def build_transform(size: int = 224, *, train: bool = False, normalize: bool = True,
                    mean: Sequence[float] = IMAGENET_MEAN,
                    std: Sequence[float] = IMAGENET_STD) -> Callable[[np.ndarray], torch.Tensor]:
    """HWC uint8 -> normalized CHW float tensor. Use for pretrained backbones
    (normalize=True). train=True adds light rotation; horizontal flip is left OFF
    (AUs can be unilateral)."""
    from torchvision.transforms import v2
    steps = [v2.ToImage(), v2.Resize((size, size), antialias=True)]
    if train:
        steps.append(v2.RandomRotation(5))
    steps.append(v2.ToDtype(torch.float32, scale=True))
    if normalize:
        steps.append(v2.Normalize(mean=tuple(mean), std=tuple(std)))
    return v2.Compose(steps)


def denormalize(t: torch.Tensor, mean: Sequence[float] = IMAGENET_MEAN,
                std: Sequence[float] = IMAGENET_STD) -> torch.Tensor:
    """Undo Normalize -> (3,H,W) in [0,1]."""
    t = t.detach().cpu()
    m = torch.tensor(tuple(mean)).view(3, 1, 1)
    s = torch.tensor(tuple(std)).view(3, 1, 1)
    return (t * s + m).clamp(0.0, 1.0)