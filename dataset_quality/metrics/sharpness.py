"""Sharpness metrics."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .preprocess import resize_short_side, resize_to


def laplacian_var(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def compute_sharpness(gray: np.ndarray, short_side: int = 256) -> dict[str, float]:
    resized = resize_short_side(gray, short=short_side)
    return {
        "lap_var": laplacian_var(resized),
        "tenengrad": tenengrad(resized),
    }


def _clip_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def compute_roi_sharpness(
    gray: np.ndarray,
    roi_bbox: Tuple[int, int, int, int],
    roi_size: Tuple[int, int] = (160, 160),
) -> dict[str, float]:
    h, w = gray.shape[:2]
    x, y, bw, bh = _clip_bbox(roi_bbox, w, h)
    roi = gray[y : y + bh, x : x + bw]
    roi = resize_to(roi, w=roi_size[0], h=roi_size[1])
    return {
        "roi_lap_var": laplacian_var(roi),
        "roi_tenengrad": tenengrad(roi),
    }
