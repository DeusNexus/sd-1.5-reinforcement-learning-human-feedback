"""Face detection and ROI helpers."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2


_CACHED_CASCADES: dict[str, cv2.CascadeClassifier] = {}


def _load_cascade(cascade_path: str | None) -> cv2.CascadeClassifier:
    path = cascade_path or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if path not in _CACHED_CASCADES:
        _CACHED_CASCADES[path] = cv2.CascadeClassifier(path)
    return _CACHED_CASCADES[path]


def detect_faces(
    gray: np.ndarray,
    cascade_path: str | None = None,
    scale_factor: float = 1.05,
    min_neighbors: int = 3,
    min_size: Tuple[int, int] = (24, 24),
) -> list[Tuple[int, int, int, int]]:
    cascade = _load_cascade(cascade_path)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def select_largest_face(
    bboxes: Iterable[Tuple[int, int, int, int]] | None,
) -> Tuple[int, int, int, int] | None:
    if not bboxes:
        return None
    best = None
    best_area = 0
    for x, y, w, h in bboxes:
        area = int(w) * int(h)
        if area > best_area:
            best_area = area
            best = (int(x), int(y), int(w), int(h))
    return best


def fallback_roi(gray: np.ndarray, ratio: float = 0.6) -> Tuple[int, int, int, int]:
    h, w = gray.shape[:2]
    rw = int(round(w * ratio))
    rh = int(round(h * ratio))
    x = max(0, (w - rw) // 2)
    y = max(0, (h - rh) // 2)
    return (x, y, max(1, rw), max(1, rh))


def face_area_ratio(bbox: Tuple[int, int, int, int] | None, shape: Tuple[int, int]) -> float:
    if bbox is None:
        return 0.0
    h, w = shape[:2]
    _, _, bw, bh = bbox
    return float(bw * bh) / float(w * h)
