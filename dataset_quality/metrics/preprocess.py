"""Shared preprocessing utilities for metrics."""

from __future__ import annotations

import cv2
import numpy as np


def _resize(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    if gray.shape[0] == height and gray.shape[1] == width:
        return gray
    interp = cv2.INTER_AREA if height < gray.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(gray, (width, height), interpolation=interp)


def resize_short_side(gray: np.ndarray, short: int = 256) -> np.ndarray:
    if gray is None or gray.size == 0:
        return gray
    h, w = gray.shape[:2]
    if min(h, w) == short:
        return gray
    scale = float(short) / float(min(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return _resize(gray, new_w, new_h)


def resize_to(gray: np.ndarray, w: int = 160, h: int = 160) -> np.ndarray:
    return _resize(gray, w, h)
