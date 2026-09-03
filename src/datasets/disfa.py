from __future__ import annotations

"""
DISFA loader. Produces the same Unit contract as painfacereader, so it slots into
preprocess / folds / AUDataset / training unchanged.

Assumed layout (the standard DISFA distribution):
    <root>/ActionUnit_Labels/SN0xx/SN0xx_au{N}.txt   # lines "frame,intensity" (0-5)
    <root>/Video_LeftCamera/*SN0xx*.avi              # one video per subject/camera
The 12 coded AUs are 1,2,4,5,6,9,12,15,17,20,25,26; detection binarizes intensity>=2.
One Unit per subject (for the chosen camera); subject = SN0xx for the folds.
"""

import warnings
from pathlib import Path

import numpy as np

from src.config import DISFAConfig
from src.datasets.utils import Unit, VideoSource

AU_IDS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
INTENSITY_THRESHOLD = 2                            # occurrence := intensity >= this


def _read_intensities(path: Path) -> np.ndarray:
    """Read a DISFA au txt (lines 'frame,intensity') -> int8 intensities in row order."""
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            vals.append(int(float(line.split(",")[-1])))
    return np.array(vals, dtype=np.int8)


def load_units(config: DISFAConfig) -> tuple[list[Unit], list[str]]:
    labels_dir = Path(config.labels)
    videos_dir = Path(config.videos)
    au_names = [f"AU{a}" for a in AU_IDS]

    units: list[Unit] = []
    for subj_dir in sorted(p for p in labels_dir.iterdir() if p.is_dir()):
        subject = subj_dir.name                    # e.g. SN001
        cols = []
        for a in AU_IDS:
            txt = subj_dir / f"{subject}_au{a}.txt"
            if not txt.exists():
                warnings.warn(f"[disfa] missing {txt.name}; skipping {subject}")
                cols = None
                break
            cols.append(_read_intensities(txt))
        if cols is None:
            continue

        n = min(len(c) for c in cols)              # align in case of ragged lengths
        labels = (np.stack([c[:n] for c in cols], axis=1) >= INTENSITY_THRESHOLD).astype(np.int8)

        videos = sorted(videos_dir.glob(f"*{subject}*.avi"))
        if not videos:
            warnings.warn(f"[disfa] no video for {subject} in {videos_dir}; skipping")
            continue

        units.append(Unit(
            unit_id=f"{subject}_{config.camera[0].upper()}",
            subject=subject,
            labels=labels,
            source=VideoSource(path=str(videos[0]), frames=list(range(n))),
            meta={"camera": config.camera},
        ))

    print(f"[disfa] {len(units)} units (camera={config.camera}), {len(au_names)} AUs")
    return units, au_names