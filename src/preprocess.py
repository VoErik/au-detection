from __future__ import annotations

"""
Offline preprocessing: turn each Unit's raw frames (video or image folder) into a
stabilized-crop stack on disk, once. Training then reads crops by array index.

All alignment lives here because it only happens at preprocess time: detect 5-point
landmarks on a few probe frames, derive ONE face box for the unit (head motion
preserved), crop every frame to it. Output: crops_dir/{unit_id}.npy, shape
(n_frames, out_size, out_size, 3) uint8.
"""

import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from src.datasets.utils import ImageSource, Unit, VideoSource

# 5-point ArcFace template (left eye, right eye, nose, left mouth, right mouth) at
# 112px -- used only for its eye->mouth proportion when sizing the box.
_TEMPLATE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                          [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def build_detector(device: str = "cuda"):
    """MTCNN, built once and threaded into preprocess()."""
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=True, device=device)


def _detect(detector, imgs: np.ndarray, *, batch_size: int = 32,
            scale: float = 1.0) -> np.ndarray:
    """(N,H,W,3) uint8 -> (N,5,2) float32 landmarks, NaN where no face. scale<1
    downscales for speed and maps coordinates back to full resolution."""
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
    eye->mouth distance scaled by the template's proportion (so `margin` frames the
    face consistently regardless of resolution)."""
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
    """Return read(rows) -> (len(rows), H, W, 3) uint8 RGB for this source. Video
    frames are read with imageio's ffmpeg backend (cross-platform incl. macOS --
    imageio-ffmpeg bundles its own ffmpeg, so no decord and no system install)."""
    if isinstance(source, VideoSource):
        import imageio.v2 as imageio
        reader = imageio.get_reader(source.path)
        return lambda rows: np.stack([np.asarray(reader.get_data(source.frames[r])) for r in rows])
    if isinstance(source, ImageSource):
        return lambda rows: np.stack([cv2.imread(source.paths[r])[:, :, ::-1] for r in rows])
    raise TypeError(f"unknown frame source: {type(source)}")


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
    chunk: int = 64,
    overwrite: bool = False,
) -> None:
    """Write crops_dir/{unit_id}.npy for every unit. Detects on n_probe evenly
    spaced frames (enough for a stable box on a seated subject), then crops all
    frames to that box in `chunk`-sized reads so peak memory stays small."""
    from tqdm.auto import tqdm
    detector = detector or build_detector(device)
    out = Path(crops_dir)
    out.mkdir(parents=True, exist_ok=True)

    for u in tqdm(units, desc="preprocess", unit="unit"):
        dst = out / f"{u.unit_id}.npy"
        if dst.exists() and not overwrite:
            continue
        n = u.n_frames
        read = _frame_reader(u.source)

        probe = np.unique(np.linspace(0, n - 1, min(n_probe, n)).astype(int))
        lm = _detect(detector, read(probe), scale=detect_scale)
        if np.isnan(lm[:, 0, 0]).all():
            warnings.warn(f"no face detected for {u.unit_id}; skipped")
            continue
        box = _video_box(lm, margin, out_size)

        crops = np.empty((n, out_size, out_size, 3), dtype=np.uint8)
        for a in range(0, n, chunk):
            rows = list(range(a, min(a + chunk, n)))
            imgs = read(rows)
            for t, r in enumerate(rows):
                crops[r] = _crop(imgs[t], box, out_size)
        np.save(dst, crops)