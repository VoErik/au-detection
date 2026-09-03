from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import json
from pathlib import Path

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


# --------------------------------------------------------------------------- #
# Subject-disjoint k-fold creation.                                           #
#                                                                             #
# Simple by design: draw many random balanced subject-partitions, score each  #
# (lexicographic: first coverage -- every kept AU has a carrier in every fold; #
# then prevalence balance across folds), keep the best. No annealing.         #
# --------------------------------------------------------------------------- #

def _subject_stats(units, au_names: list[str]):
    """Aggregate per subject: active-frame counts per AU (S,K) and total frames (S,)."""
    active: dict[str, np.ndarray] = {}
    total: dict[str, int] = {}
    for u in units:
        s = u.subject
        if s not in active:
            active[s] = u.labels.sum(axis=0).astype(float)
            total[s] = u.labels.shape[0]
        else:
            active[s] += u.labels.sum(axis=0)
            total[s] += u.labels.shape[0]
    subjects = sorted(active)
    A = np.stack([active[s] for s in subjects])
    T = np.array([total[s] for s in subjects], dtype=float)
    return subjects, A, T


def filter_aus(units, au_names: list[str], *, min_prevalence: float = 0.02,
               min_subjects: int = 8, min_carrier_frames: int = 25) -> list[str]:
    """Keep AUs that are both frequent enough overall AND carried by enough distinct
    subjects (so they can be stratified and are likely to generalise)."""
    _, A, T = _subject_stats(units, au_names)
    prevalence = A.sum(axis=0) / T.sum()
    n_carriers = (A >= min_carrier_frames).sum(axis=0)
    keep = (prevalence >= min_prevalence) & (n_carriers >= min_subjects)
    return [a for a, k in zip(au_names, keep) if k]


def _random_partition(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(n)
    fold = np.empty(n, dtype=int)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    start = 0
    for f, s in enumerate(sizes):
        fold[order[start:start + s]] = f
        start += s
    return fold


def _fold_state(fold, contrib, carrier, k):
    """Per-fold aggregates the annealer keeps in sync across swaps."""
    K = contrib.shape[1]
    sum_c = np.zeros((k, K))
    cc = np.zeros((k, K), dtype=int)
    np.add.at(sum_c, fold, contrib)
    np.add.at(cc, fold, carrier)
    size = np.bincount(fold, minlength=k)
    return sum_c, cc, size


def _energy(sum_c, cc, size) -> tuple[int, float]:
    """(coverage_missing, prevalence_cv) -- compared as a tuple, so coverage
    dominates balance lexicographically."""
    missing = int((cc == 0).sum())
    S = sum_c / size[:, None]
    m = S.mean(axis=0)
    balance = float((S.std(axis=0) / (m + 1e-9)).sum())
    return missing, balance


def make_folds(units, au_names: list[str], kept_aus: list[str], *, k: int = 5,
               min_carrier_frames: int = 25, n_restarts: int = 8, n_iter: int = 6000,
               t0: float = 0.05, t_min: float = 1e-4, seed: int = 0) -> list[list[str]]:
    """Assign subjects to k person-disjoint folds via multi-restart simulated
    annealing over subject swaps. Objective is lexicographic: first coverage
    (every kept AU has a carrier in every fold -- a hard gate the search never
    trades away), then prevalence balance across folds. Returns subject lists."""
    missing_aus = [a for a in kept_aus if a not in au_names]
    if missing_aus:
        raise ValueError(f"AUs not in this dataset's AU set: {missing_aus}. Available: {au_names}")
    subjects, A, T = _subject_stats(units, au_names)
    idx = [au_names.index(a) for a in kept_aus]
    A = A[:, idx]
    prev = A / T[:, None]
    carrier = (A >= min_carrier_frames).astype(int)          # int: bool '-' is unsupported
    S = len(subjects)

    infeasible = [a for a, c in zip(kept_aus, carrier.sum(axis=0)) if c < k]
    if infeasible:
        import warnings
        warnings.warn(f"AUs with < k={k} carriers can't cover every fold: {infeasible}")

    rng = np.random.default_rng(seed)
    best_fold, best_e = None, (np.inf, np.inf)
    for _ in range(n_restarts):
        fold = _random_partition(S, k, rng)
        sc, cc, size = _fold_state(fold, prev, carrier, k)
        cur = _energy(sc, cc, size)
        if cur < best_e:
            best_e, best_fold = cur, fold.copy()
        for t in range(n_iter):
            temp = t0 * (t_min / t0) ** (t / max(n_iter - 1, 1))
            i, j = rng.integers(0, S, size=2)
            if fold[i] == fold[j]:
                continue
            fi, fj = int(fold[i]), int(fold[j])
            sc_fi, sc_fj, cc_fi, cc_fj = sc[fi].copy(), sc[fj].copy(), cc[fi].copy(), cc[fj].copy()
            sc[fi] += prev[j] - prev[i]; sc[fj] += prev[i] - prev[j]
            cc[fi] += carrier[j] - carrier[i]; cc[fj] += carrier[i] - carrier[j]
            cand = _energy(sc, cc, size)
            (cov, bal), (cov2, bal2) = cur, cand
            if cov2 < cov:
                accept = True
            elif cov2 > cov:
                accept = False
            elif bal2 <= bal:
                accept = True
            else:
                accept = rng.random() < np.exp(-(bal2 - bal) / temp)
            if accept:
                fold[i], fold[j] = fj, fi
                cur = cand
                if cand < best_e:
                    best_e, best_fold = cand, fold.copy()
            else:
                sc[fi], sc[fj], cc[fi], cc[fj] = sc_fi, sc_fj, cc_fi, cc_fj
    return [[subjects[i] for i in range(S) if best_fold[i] == f] for f in range(k)]


def cv_splits(folds: list[list[str]]):
    """Yield {'fold','train','val','test'} -- fold i test, (i+1)%k val, rest train."""
    k = len(folds)
    for i in range(k):
        yield {
            "fold": i,
            "test": folds[i],
            "val": folds[(i + 1) % k],
            "train": [s for f in range(k) if f not in (i, (i + 1) % k) for s in folds[f]],
        }


def assert_no_leakage(folds: list[list[str]]) -> None:
    """Raise if any subject appears in more than one fold."""
    seen: dict[str, int] = {}
    for f, people in enumerate(folds):
        for p in people:
            if p in seen:
                raise AssertionError(f"LEAKAGE: {p} in folds {seen[p]} and {f}")
            seen[p] = f


def save_folds(path, folds: list[list[str]], kept_aus: list[str], *, meta: dict | None = None) -> None:
    """Persist a split reproducibly: the subject folds + the AU list they were built
    for (+ optional meta like seed / k / filter params). JSON."""
    Path(path).write_text(json.dumps(
        {"folds": [list(f) for f in folds], "kept_aus": list(kept_aus), "meta": meta or {}},
        indent=2))


def load_folds(path) -> tuple[list[list[str]], list[str], dict]:
    """Load a split saved by save_folds -> (folds, kept_aus, meta)."""
    d = json.loads(Path(path).read_text())
    return d["folds"], d["kept_aus"], d.get("meta", {})


def subset_units(units, au_names: list[str], kept_aus: list[str]):
    """Return units with labels sliced to kept_aus (in that order), so their label
    columns line up with the au_names you hand to AUDataset. load_units gives full-
    AU-set labels; call this once after choosing kept_aus."""
    missing = [a for a in kept_aus if a not in au_names]
    if missing:
        raise ValueError(f"AUs not in au_names: {missing}")
    idx = [au_names.index(a) for a in kept_aus]
    return [Unit(u.unit_id, u.subject, u.labels[:, idx].copy(),
                 source=u.source, meta=dict(u.meta)) for u in units]