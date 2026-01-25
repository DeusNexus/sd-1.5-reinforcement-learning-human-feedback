"""JPEG blockiness metrics."""

from __future__ import annotations

import numpy as np


def _mean_or_zero(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def jpeg_blockiness(gray: np.ndarray, block_size: int = 8) -> float:
    h, w = gray.shape[:2]
    if h < block_size * 2 or w < block_size * 2:
        return 0.0

    gray = gray.astype(np.float32)
    dv = np.abs(gray[:, 1:] - gray[:, :-1])
    dh = np.abs(gray[1:, :] - gray[:-1, :])

    cols = np.arange(1, w)
    rows = np.arange(1, h)
    v_boundary = (cols % block_size) == 0
    h_boundary = (rows % block_size) == 0

    vb = _mean_or_zero(dv[:, v_boundary])
    vnb = _mean_or_zero(dv[:, ~v_boundary])
    hb = _mean_or_zero(dh[h_boundary, :])
    hnb = _mean_or_zero(dh[~h_boundary, :])

    return float((vb - vnb) + (hb - hnb))


def artifact_score(blockiness: float, tenengrad: float, eps: float = 1e-6) -> float:
    return float(blockiness / (tenengrad + eps))
