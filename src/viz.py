from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from src.utils import IMAGENET_MEAN, IMAGENET_STD, denormalize


def _to_uint8(x: torch.Tensor, normalized: bool,
              mean: Sequence[float] = IMAGENET_MEAN,
              std: Sequence[float] = IMAGENET_STD) -> np.ndarray:
    t = denormalize(x, mean, std) if normalized else x.detach().cpu().clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _active(au_names: Sequence[str], row) -> str:
    return ", ".join(a for a, v in zip(au_names, row) if float(v) > 0.5) or "—"


def show_sample(dataset, i: int, *, normalized: bool = False, max_frames: int = 8):
    """Visualise dataset[i]. Frame unit -> a single crop; window unit -> an
    evenly-spaced filmstrip. Title shows the target frame's active AUs. Set
    normalized=True if the dataset's transform normalizes (pretrained backbones)."""
    import matplotlib.pyplot as plt
    item = dataset[i]
    x, y = item[0], item[1]
    au = dataset.au_names

    if x.ndim == 3:                                        # frame
        fig, ax = plt.subplots(figsize=(3.2, 3.4))
        ax.imshow(_to_uint8(x, normalized))
        ax.set_title(_active(au, y), fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        return fig

    T = x.shape[0]                                         # window
    picks = np.linspace(0, T - 1, min(max_frames, T)).astype(int)
    fig, axes = plt.subplots(1, len(picks), figsize=(1.9 * len(picks), 2.4))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, picks):
        ax.imshow(_to_uint8(x[t], normalized))
        ax.set_title(f"t={t}", fontsize=7)
        ax.axis("off")
    fig.suptitle("target AUs: " + _active(au, y), fontsize=9)
    fig.tight_layout()
    return fig