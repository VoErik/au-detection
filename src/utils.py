from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

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


# --------------------------------------------------------------------------- #
# Logging, checkpoints, experiment artifacts                                  #
# --------------------------------------------------------------------------- #

import json
import logging
from datetime import datetime
from pathlib import Path


class Logger:
    """Thin wrapper over logging: console + optional file, no duplicate handlers."""

    def __init__(self, name: str, level: int = logging.INFO, logfile=None):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(level)
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)
            if logfile is not None:
                fh = logging.FileHandler(logfile)
                fh.setFormatter(fmt)
                self.logger.addHandler(fh)
        self.logger.propagate = False

    def get_logger(self) -> logging.Logger:
        return self.logger


def save_model(model, path) -> None:
    """Save a model's weights (state_dict)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, *, map_location="cpu"):
    """Load weights (state_dict) into an already-built model, in place."""
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model


class Experiment:
    """One experiment = one directory holding every artifact: the exact config used
    (config.yaml), a run.log, model checkpoints, metrics, and figures.

        exp = Experiment(cfg, name="pfr_baseline")
        log = exp.log                       # logs to console AND exp.dir/run.log
        exp.save_model(model, "model_fold0.pt")
        exp.save_metrics(df, summary)
        exp.save_figure(fig, "au_f1.png")
    """

    def __init__(self, config, root: str = "runs", name: Optional[str] = None):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = name or f"{config.task}_{config.model}"
        self.dir = Path(root) / f"{label}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        config.to_yaml(self.dir / "config.yaml")               # copy the config used
        self.log = Logger(label, logfile=self.dir / "run.log").get_logger()
        self.log.info(f"experiment dir: {self.dir}")

    def path(self, *parts) -> Path:
        return self.dir.joinpath(*parts)

    def save_model(self, model, name: str = "model.pt") -> Path:
        p = self.dir / name
        save_model(model, p)
        return p

    def save_metrics(self, df, summary: dict) -> None:
        df.to_csv(self.dir / "metrics.csv")
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def save_figure(self, fig, name: str = "figure.png") -> Path:
        p = self.dir / name
        fig.savefig(p, dpi=120, bbox_inches="tight")
        return p


def get_param_groups_llrd(model, base_lr: float, weight_decay: float, layer_decay: float = 0.65):
    """Groups ViT parameters by depth for Layer-wise LR Decay."""
    num_layers = len(model.backbone.blocks) if hasattr(model, 'backbone') else 12
    param_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if name.startswith("backbone.patch_embed") or name.startswith("backbone.cls_token") or name.startswith("backbone.pos_embed"):
            layer_id = 0
        elif name.startswith("backbone.blocks."):
            layer_id = int(name.split(".")[2]) + 1
        else: # fc_norm and head
            layer_id = num_layers + 1
            
        group_lr = base_lr * (layer_decay ** (num_layers + 1 - layer_id))
        group_wd = 0.0 if len(param.shape) == 1 or name.endswith(".bias") else weight_decay
        
        group_name = f"layer_{layer_id}_wd_{group_wd}"
        if group_name not in param_groups:
            param_groups[group_name] = {"params": [], "lr": group_lr, "weight_decay": group_wd}
            
        param_groups[group_name]["params"].append(param)
        
    return list(param_groups.values())