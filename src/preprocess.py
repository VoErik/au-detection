from __future__ import annotations

"""
Offline preprocessing: turn each Unit's raw frames (video or image folder) into a
stabilized-crop stack on disk, once. Training then reads crops by array index.

Two phases, because the costs are different:
  1. detect a box per unit  -- GPU, tiny (only n_probe frames/unit);
  2. crop every frame        -- CPU/decode bound, the slow part -> run in parallel
     across units, decoding each video sequentially (no per-frame seeking, and we
     stop at the last annotated frame instead of walking the whole full-length file).

Output: crops_dir/{unit_id}.npy, shape (n_frames, out_size, out_size, 3) uint8.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from src.datasets.utils import ImageSource, Unit, VideoSource

_TEMPLATE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                          [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def build_detector(device: str = "cuda"):
    """MTCNN, built once and threaded into preprocess()."""
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=True, device=device)


def _detect(detector, imgs: np.ndarray, *, batch_size: int = 32,
            scale: float = 1.0) -> np.ndarray:
    """(N,H,W,3) uint8 -> (N,5,2) float32 landmarks, NaN where no face."""
    N, H, W = imgs.shape[:3]
    lm = np.full((N, 5, 2), np.nan, dtype=np.float32)
    frames, sx, sy = imgs, 1.0, 1.0
    if scale != 1.0:
        newW, newH = max(1, round(W * scale)), max(1, round(H * scale))
        frames = np.stack([cv2.resize(im, (newW, newH), interpolation=cv2.INTER_AREA)
                           for im in imgs])
        sx, sy = W / newW, H / newH
    for b in range(0, N, batch_size):
        chunk = frames[b:b + batch_size]
        boxes, _, points = detector.detect(chunk, landmarks=True)
        for k in range(len(chunk)):
            bx, pts = boxes[k], points[k]
            if pts is None or len(pts) == 0:
                continue
            areas = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
            p = np.asarray(pts[int(np.argmax(areas))], dtype=np.float32)
            p[:, 0] *= sx
            p[:, 1] *= sy
            lm[b + k] = p
    return lm


def _video_box(landmarks: np.ndarray, margin: float, out_size: int) -> tuple[int, int, int, int]:
    """One square (x0,y0,x1,y1) box for a whole unit, sized from the average
    eye->mouth distance scaled by the template's proportion."""
    valid = landmarks[~np.isnan(landmarks[:, 0, 0])]
    if len(valid) == 0:
        raise ValueError("no valid landmarks")
    centre = valid.reshape(-1, 2).mean(axis=0)
    eye_c = valid[:, :2, :].mean(axis=1)
    mouth_c = valid[:, 3:, :].mean(axis=1)
    eye_to_mouth = np.linalg.norm(mouth_c - eye_c, axis=1).mean()
    t_eye, t_mouth = _TEMPLATE_112[:2].mean(0), _TEMPLATE_112[3:].mean(0)
    frac = np.linalg.norm(t_mouth - t_eye) / 112.0
    half = max((1.0 + margin) * eye_to_mouth / frac / 2.0, 1.0)
    cx, cy = centre
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def _crop(img: np.ndarray, box: tuple[int, int, int, int], out_size: int) -> np.ndarray:
    """Crop to box (clamped + edge-padded if it runs off the frame) and resize."""
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    px0, py0, px1, py1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    crop = img[py0:py1, px0:px1]
    pad = [(py0 - y0, y1 - py1), (px0 - x0, x1 - px1), (0, 0)]
    if any(p for pair in pad for p in pair):
        crop = np.pad(crop, pad, mode="edge")
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _frame_reader(source) -> Callable[[Sequence[int]], np.ndarray]:
    """Return read(rows) -> (len(rows), H, W, 3) uint8 RGB. Video frames via
    imageio's ffmpeg backend (cross-platform incl. macOS; no decord). Used for the
    sparse probe read; the crop pass decodes sequentially (see _crop_unit)."""
    if isinstance(source, VideoSource):
        import imageio.v2 as imageio
        reader = imageio.get_reader(source.path)
        return lambda rows: np.stack([np.asarray(reader.get_data(source.frames[r])) for r in rows])
    if isinstance(source, ImageSource):
        return lambda rows: np.stack([cv2.imread(source.paths[r])[:, :, ::-1] for r in rows])
    raise TypeError(f"unknown frame source: {type(source)}")


def _crop_unit(args) -> tuple[str, int]:
    """Worker: decode a unit's frames sequentially, crop each to its box, save.
    Returns (unit_id, n_unfilled) so the caller can flag units with missing frames."""
    unit, box, crops_dir, out_size = args
    src = unit.source
    n = unit.n_frames
    crops = np.zeros((n, out_size, out_size, 3), dtype=np.uint8)
    filled = 0
    if isinstance(src, VideoSource):
        import imageio.v2 as imageio
        want: dict[int, int] = {}
        for r, f in enumerate(src.frames):
            want.setdefault(f, r)                       # video-frame index -> label row
        last = max(src.frames)
        reader = imageio.get_reader(src.path)
        try:
            for i, frame in enumerate(reader):          # sequential decode, no seeking
                r = want.get(i)
                if r is not None:
                    crops[r] = _crop(np.asarray(frame), box, out_size)
                    filled += 1
                if i >= last:                           # don't walk the rest of the video
                    break
        finally:
            reader.close()
    else:
        for r, p in enumerate(src.paths):
            crops[r] = _crop(cv2.imread(p)[:, :, ::-1], box, out_size)
            filled += 1
    np.save(Path(crops_dir) / f"{unit.unit_id}.npy", crops)
    return unit.unit_id, n - filled


def preprocess(
    units: Sequence[Unit],
    crops_dir: str,
    *,
    out_size: int = 224,
    margin: float = 0.15,
    detector=None,
    device: str = "cuda",
    n_probe: int = 32,
    detect_scale: float = 0.5,
    batch_size: int = 32,
    n_workers: int | None = None,
    overwrite: bool = False,
) -> None:
    """Write crops_dir/{unit_id}.npy for every unit.

    Phase 1 (GPU, cheap): detect landmarks on n_probe frames/unit -> one box.
    Phase 2 (CPU, the slow part): crop every frame, parallelised across units with
    n_workers processes (default os.cpu_count()). batch_size only affects the small
    detection phase; n_workers is the lever that matters for wall-time.
    """
    from tqdm.auto import tqdm
    out = Path(crops_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [u for u in units if overwrite or not (out / f"{u.unit_id}.npy").exists()]
    if not todo:
        print(f"[preprocess] all {len(units)} crops present; nothing to do")
        return

    # ---- phase 1: a box per unit (GPU) ----
    detector = detector or build_detector(device)
    jobs = []
    for u in tqdm(todo, desc="detect boxes", unit="unit"):
        read = _frame_reader(u.source)
        probe = np.unique(np.linspace(0, u.n_frames - 1, min(n_probe, u.n_frames)).astype(int))
        lm = _detect(detector, read(probe), batch_size=batch_size, scale=detect_scale)
        if np.isnan(lm[:, 0, 0]).all():
            warnings.warn(f"no face detected for {u.unit_id}; skipped")
            continue
        jobs.append((u, _video_box(lm, margin, out_size), str(out), out_size))

    # ---- phase 2: crop all frames (CPU, parallel across units) ----
    n_workers = n_workers or os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for unit_id, missing in tqdm(ex.map(_crop_unit, jobs), total=len(jobs),
                                     desc=f"crop frames (x{n_workers})", unit="unit"):
            if missing:
                warnings.warn(f"{unit_id}: {missing} frame(s) not found in video; left blank")