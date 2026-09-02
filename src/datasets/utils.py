from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Windowing: one input -> one target frame's label, for every model.          #
#   causal   -> window [n-L+1 .. n]  (target = last frame; real-time)          #
#   centered -> window [n-L//2 .. n-L//2+L-1]  (target at offset L//2)         #
# window_len=1 -> every frame is valid (the frame-model case).                 #
# --------------------------------------------------------------------------- #

def window_indices(target: int, window_len: int, mode: str = "causal") -> list[int]:
    start = target - window_len + 1 if mode == "causal" else target - window_len // 2
    return list(range(start, start + window_len))


def valid_target_frames(n_frames: int, window_len: int, mode: str = "causal") -> list[int]:
    """Target frames whose whole window fits inside [0, n_frames)."""
    out = []
    for n in range(n_frames):
        w = window_indices(n, window_len, mode)
        if w[0] >= 0 and w[-1] <= n_frames - 1:
            out.append(n)
    return out


# --------------------------------------------------------------------------- #
# Raw-frame sources -- used ONLY by preprocessing, to abstract video vs images.#
# --------------------------------------------------------------------------- #

@dataclass
class VideoSource:
    """Read frames from a video file at the given frame indices (PainFaceReader, DISFA)."""
    path: str
    frames: list[int]                    # one video-frame index per label row


@dataclass
class ImageSource:
    """Read frames from a list of image files, one per label row (BP4D)."""
    paths: list[str]


FrameSource = Union[VideoSource, ImageSource]


# --------------------------------------------------------------------------- #
# The universal record every dataset produces.                                #
# --------------------------------------------------------------------------- #

@dataclass
class Unit:
    unit_id: str                         # unique; the crop file stem
    subject: str                         # for subject-disjoint folds
    labels: np.ndarray                   # (n_frames, K) int8 binary AU occurrence
    source: Optional[FrameSource] = None # how to read raw frames (preprocess only)
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.labels.shape[0]


def binarize(intensity: np.ndarray, thresh: int = 2) -> np.ndarray:
    """Ordinal AU intensity (0..5) -> binary occurrence at >= thresh. The standard
    DISFA detection convention is thresh=2."""
    return (np.asarray(intensity) >= thresh).astype(np.int8)


def read_csv_smart(path, sep: str = ",") -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"could not decode {path}")