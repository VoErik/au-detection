from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PainfaceReaderConfig
from src.datasets.utils import Unit, VideoSource, read_csv_smart

N_ANNOT_ROWS = 351                       # annotated rows per PFR video

_PERSON_RE = re.compile(r"[A-Za-z]{2,4}\d{4,8}")
_ANNOT_RE = re.compile(r"([A-Za-z]{2,4}\d{4,8})_(\d{2,4})", re.IGNORECASE)
_CODE_TOKEN_RE = re.compile(r"^[A-Za-z]\.\d+$")
_NON_AU = (re.compile(r"^\s*time\s*$", re.I), re.compile(r"^\s*led", re.I),
           re.compile(r"^\s*frame", re.I), re.compile(r"^unnamed", re.I))


def _parse_video_name(name: str) -> tuple[str | None, str | None, str | None]:
    """'AAA010101_frontal_druck_no_pain_D.04.mp4' -> (view, stimulus, condition)."""
    stem = name[:-4] if name.lower().endswith(".mp4") else name
    view = ("frontal" if "frontal" in stem.lower()
            else "lateral" if "lateral" in stem.lower() else None)
    body = stem.split("_")[1:]
    if body and _CODE_TOKEN_RE.match(body[-1]):
        body = body[:-1]
    if body and body[0].lower() in ("frontal", "lateral"):
        body = body[1:]
    stimulus = body[0] if body else None
    condition = "_".join(body[1:]) if len(body) > 1 else None
    return view, stimulus, condition


def _annotated_people(root: Path) -> set[str]:
    df = pd.read_csv(root / "vpcodes_n40.csv")
    col = "vp_code" if "vp_code" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip())


def _load_key(videos_dir: Path, person: str, cache: dict) -> pd.DataFrame | None:
    if person not in cache:
        hits = list(videos_dir.glob(f"{person}*odierung*.csv"))
        if not hits:
            cache[person] = None
        else:
            df = read_csv_smart(hits[0], sep=";")
            df.columns = [c.strip() for c in df.columns]
            cache[person] = df
    return cache[person]


def _au_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not any(p.match(str(c)) for p in _NON_AU)]


def _labels(df: pd.DataFrame, au_names: list[str]) -> np.ndarray:
    vals = df.reindex(columns=au_names).apply(pd.to_numeric, errors="coerce").fillna(0).values
    binary = (vals != 0).astype(np.int8)
    if binary.shape[0] != N_ANNOT_ROWS:                       # clip / pad to a fixed length
        fixed = np.zeros((N_ANNOT_ROWS, binary.shape[1]), np.int8)
        n = min(N_ANNOT_ROWS, binary.shape[0])
        fixed[:n] = binary[:n]
        binary = fixed
    return binary


def load_units(config: PainfaceReaderConfig) -> tuple[list[Unit], list[str]]:
    """Build one Unit per coded video matching config.view, plus the AU name list.
    Labels come from the *_timeline.csv; view/stimulus/condition from the per-person
    Kodierungsschlüssel. Frames are read later by preprocess via the VideoSource."""
    from tqdm.auto import tqdm
    root = Path(config.root)
    videos_dir = Path(config.videos)
    annot_dir = Path(config.annotations)
    annotated = _annotated_people(root)

    au_names: list[str] | None = None
    key_cache: dict = {}
    units: list[Unit] = []
    n_unknown_view = n_wrong_view = 0

    for ann_path in tqdm(sorted(annot_dir.rglob("*_timeline.csv")), desc="Loading Samples", unit="People"):
        m = _ANNOT_RE.search(ann_path.name)
        if not m:
            warnings.warn(f"un-parseable annotation filename: {ann_path.name}")
            continue
        person, ordinal = m.group(1).upper(), m.group(2)
        if person not in annotated:
            continue
        coded = f"{person}_{ordinal}"
        video_path = videos_dir / person / f"{coded}.mp4"

        view = stimulus = condition = None
        key = _load_key(videos_dir, person, key_cache)
        if key is not None:
            hit = key.loc[key["kodiert"].astype(str).str.strip() == f"{coded}.mp4"]
            if not hit.empty:
                view, stimulus, condition = _parse_video_name(str(hit.iloc[0]["unkodiert"]))
        if view is None:
            n_unknown_view += 1
            continue
        if view != config.view:
            n_wrong_view += 1
            continue

        df = read_csv_smart(ann_path, sep=";")
        df.columns = [str(c).strip() for c in df.columns]
        if au_names is None:
            au_names = _au_columns(df)
        labels = _labels(df, au_names)
        frames = list(range(labels.shape[0]))                 # annotation row i -> video frame i
        units.append(Unit(coded, person, labels,
                          source=VideoSource(str(video_path), frames),
                          meta={"view": view, "stimulus": stimulus, "condition": condition}))

    print(f"[painfacereader] {len(units)} units (view={config.view}); "
          f"dropped {n_wrong_view} wrong-view, {n_unknown_view} unknown-view")
    return units, (au_names or [])