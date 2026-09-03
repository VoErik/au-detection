from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from src.utils import IMAGENET_MEAN, IMAGENET_STD, denormalize

DEFAULT_CMAP = "viridis"


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


def show_fold_balance(units, au_names, folds, kept_aus, *, cmap: str = DEFAULT_CMAP):
    """Grouped bars: mean per-subject AU prevalence in each fold. Low spread per AU
    = well-stratified split; a lone tall bar = one expressive subject in that fold."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from src.datasets.utils import _subject_stats
    subjects, A, T = _subject_stats(units, au_names)
    idx = [au_names.index(a) for a in kept_aus]
    prev = A[:, idx] / T[:, None]
    sub_to_fold = {s: f for f, people in enumerate(folds) for s in people}
    k = len(folds)

    fold_prev = np.zeros((k, len(kept_aus)))
    for f in range(k):
        mask = np.array([sub_to_fold.get(s) == f for s in subjects])
        fold_prev[f] = prev[mask].mean(0) if mask.any() else 0.0

    x = np.arange(len(kept_aus))
    w = 0.8 / k
    colors = colormaps[cmap](np.linspace(0, 1, k))
    fig, ax = plt.subplots(figsize=(1.1 * len(kept_aus) + 2, 4))
    for f in range(k):
        ax.bar(x + f * w, fold_prev[f], w, label=f"fold{f}", color=colors[f])
    ax.set_xticks(x + w * (k - 1) / 2)
    ax.set_xticklabels(kept_aus, rotation=45, ha="right")
    ax.set_ylabel("mean person-prevalence")
    ax.set_title("AU prevalence across folds")
    ax.legend(fontsize=7, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


def show_au_f1(agg_df, *, cmap: str = DEFAULT_CMAP):
    """Per-AU F1 (mean +/- std over folds) as a bar chart, from aggregate_folds's
    DataFrame. Sorted best-first."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    d = agg_df.sort_values("f1_mean", ascending=False)
    aus = list(d.index)
    m = d["f1_mean"].values
    s = d["f1_std"].values
    colors = colormaps[cmap](np.linspace(0.15, 0.85, len(aus)))
    fig, ax = plt.subplots(figsize=(1.0 * len(aus) + 2, 4))
    ax.bar(np.arange(len(aus)), m, yerr=s, color=colors, capsize=3)
    ax.set_xticks(range(len(aus)))
    ax.set_xticklabels(aus, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (mean ± std over folds)")
    ax.set_title("Per-AU F1")
    fig.tight_layout()
    return fig


def show_training_curve(history, *, cmap: str = DEFAULT_CMAP):
    """Train loss (left axis) and val macro-F1 (right axis) vs step, from a fit()
    history. Marks the best-val step -- the checkpoint that gets reported."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    steps = [h["step"] for h in history]
    loss = [h["train_loss"] for h in history]
    vf1 = [h["val_macro_f1"] for h in history]
    c = colormaps[cmap]([0.15, 0.75])

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, loss, color=c[0], marker="o", ms=3, label="train loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss", color=c[0])
    ax1.tick_params(axis="y", labelcolor=c[0])

    ax2 = ax1.twinx()
    ax2.plot(steps, vf1, color=c[1], marker="s", ms=3, label="val macro-F1")
    ax2.set_ylabel("val macro-F1", color=c[1])
    ax2.tick_params(axis="y", labelcolor=c[1])
    ax2.set_ylim(0, 1)
    best = int(np.argmax(vf1))
    ax2.axvline(steps[best], ls="--", color=c[1], alpha=0.5)
    ax2.annotate(f"best {vf1[best]:.3f} @ {steps[best]}", (steps[best], vf1[best]),
                 fontsize=8, color=c[1])
    ax1.set_title("Training curve")
    fig.tight_layout()
    return fig