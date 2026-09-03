from __future__ import annotations

"""
BP4D loader. Same Unit contract as the others.

Assumed layout (the standard BP4D-Spontaneous 2D + AU-occurrence distribution):
    <root>/AUCoding/<subject>_<task>.csv     # first col = frame number, then per-AU 0/1/9
    <root>/<subject>/<task>/<frame>.jpg      # one image per frame (any zero-padding)
The 12 coded AUs are 1,2,4,6,7,10,12,14,15,17,23,24; occurrence is 1 (0/9 -> absent).
One Unit per subject/task; subject = F0xx / M0xx for the folds. CSV frame numbers are
matched to the actual image files by their numeric part, so zero-padding doesn't matter.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BP40DConfig
from src.datasets.utils import ImageSource, Unit, read_csv_smart

AU_IDS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]


def _digits(text: str) -> int | None:
    d = "".join(ch for ch in str(text) if ch.isdigit())
    return int(d) if d else None


def _au_columns(columns) -> dict[int, str]:
    """Map AU id -> the CSV column whose numeric part equals it (handles 'AU1'/'1'/'au01')."""
    out = {}
    for c in columns:
        d = _digits(c)
        if d in AU_IDS:
            out[d] = c
    return out


def load_units(config: BP40DConfig) -> tuple[list[Unit], list[str]]:
    au_coding = Path(config.au_coding)
    images_root = Path(config.images)
    au_names = [f"AU{a}" for a in AU_IDS]

    units: list[Unit] = []
    for csv_path in sorted(au_coding.glob("*.csv")):
        stem = csv_path.stem                       # e.g. F001_T1
        if "_" not in stem:
            continue
        subject, task = stem.split("_", 1)

        df = read_csv_smart(csv_path, sep=",")
        frame_col = df.columns[0]
        frames = [int(f) for f in pd.to_numeric(df[frame_col], errors="coerce").fillna(-1)]
        colmap = _au_columns(df.columns)

        labels = np.zeros((len(df), len(AU_IDS)), dtype=np.int8)
        for j, a in enumerate(AU_IDS):
            col = colmap.get(a)
            if col is not None:
                v = pd.to_numeric(df[col], errors="coerce").fillna(0).values
                labels[:, j] = (v == 1).astype(np.int8)   # 1 present; 0 and 9(unknown) -> absent

        # match CSV frame numbers to actual image files (robust to zero-padding)
        img_dir = images_root / subject / task
        avail = {}
        for p in img_dir.glob("*.jpg"):
            d = _digits(p.stem)
            if d is not None:
                avail[d] = str(p)
        keep = [i for i, f in enumerate(frames) if f in avail]
        if not keep:
            warnings.warn(f"[bp4d] no images matched for {stem} in {img_dir}; skipping")
            continue

        units.append(Unit(
            unit_id=f"{subject}_{task}",
            subject=subject,
            labels=labels[keep],
            source=ImageSource(paths=[avail[frames[i]] for i in keep]),
            meta={"task": task},
        ))

    print(f"[bp4d] {len(units)} units, {len(au_names)} AUs")
    return units, au_names