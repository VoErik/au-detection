from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.utils import Unit, valid_target_frames, window_indices
from src.utils import to_chw_float


class AUDataset(Dataset):
    """
    Dataset-agnostic reader over preprocessed crops. Each unit has a crop stack at
    crops_dir/{unit_id}.npy of shape (n_frames, S, S, 3) uint8 (memmapped) and a
    binary label array (n_frames, K). Works the same for any dataset -- the
    per-dataset code only builds the Units and the crops.

      unit_type='frame'  -> x=(3,S,S),     y=(K,)   for one target frame
      unit_type='window' -> x=(L,3,S,S),   y=(K,)   the target frame's L-window

    Both index over each unit's window-valid target frames, so a frame- and a
    window-dataset built with the same window_len/mode score exactly the same
    frames per unit. Use window_len=1 for a plain per-frame run (all frames valid).
    """

    def __init__(self, units: Sequence[Unit], crops_dir, au_names: Sequence[str], *,
                 unit_type: str = "frame", window_len: int = 16,
                 window_mode: str = "causal", transform: Optional[Callable] = None,
                 return_meta: bool = False, max_open: int = 64):
        if unit_type not in ("frame", "window"):
            raise ValueError(f"unit_type must be frame|window, got {unit_type!r}")
        if window_mode not in ("causal", "centered"):
            raise ValueError(f"window_mode must be causal|centered, got {window_mode!r}")

        self.units = list(units)
        self.crops_dir = Path(crops_dir)
        self.au_names = list(au_names)
        self.unit_type = unit_type
        self.window_len = window_len
        self.window_mode = window_mode
        self.transform = transform or to_chw_float
        self.return_meta = return_meta
        self._mm: OrderedDict = OrderedDict()   # LRU cache of open memmaps
        self._max_open = max_open

        K = len(self.au_names)
        bad = [u.unit_id for u in self.units if u.labels.shape[1] != K]
        if bad:
            raise ValueError(
                f"{len(bad)} unit(s) have labels with != {K} columns while au_names has "
                f"{K} (e.g. {bad[:3]}). Slice labels first: subset_units(units, au_names, kept).")

        rows = []
        for ui, u in enumerate(self.units):
            for n in valid_target_frames(u.n_frames, window_len, window_mode):
                rows.append((ui, n))
        self._index = np.asarray(rows, dtype=np.int64).reshape(-1, 2)

    @property
    def n_classes(self) -> int:
        return len(self.au_names)

    def __len__(self) -> int:
        return len(self._index)

    def _crops(self, u: Unit) -> np.ndarray:
        mm = self._mm.get(u.unit_id)
        if mm is None:
            mm = np.load(self.crops_dir / f"{u.unit_id}.npy", mmap_mode="r")
            self._mm[u.unit_id] = mm
            while len(self._mm) > self._max_open:            # evict LRU, free its fd
                _, old = self._mm.popitem(last=False)
                obj = getattr(old, "_mmap", None)
                if obj is not None:
                    obj.close()
        else:
            self._mm.move_to_end(u.unit_id)                  # mark most-recently-used
        return mm

    def __getitem__(self, i: int):
        ui, pos = (int(v) for v in self._index[i])
        u = self.units[ui]
        crops = self._crops(u)
        if self.unit_type == "frame":
            x = self.transform(np.array(crops[pos]))
        else:
            idxs = window_indices(pos, self.window_len, self.window_mode)
            x = torch.stack([self.transform(np.array(crops[j])) for j in idxs], dim=0)
        y = torch.from_numpy(u.labels[pos].astype(np.float32))
        if not self.return_meta:
            return x, y
        return x, y, {"unit": u.unit_id, "subject": u.subject, "pos": pos, **u.meta}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = OrderedDict()            # reopen memmaps lazily per worker
        return state