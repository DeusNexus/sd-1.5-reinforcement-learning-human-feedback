"""NR-IQA metrics (NIQE/BRISQUE) with capability detection."""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

from .preprocess import resize_short_side

_LOGGER = logging.getLogger(__name__)
_WARNED = {"niqe": False, "brisque": False}
_PYIQA_NIQE = None


def supports_opencv_quality() -> bool:
    return hasattr(cv2, "quality")


def _warn_once(key: str, message: str) -> None:
    if mp.current_process().name != "MainProcess":
        return
    if not _WARNED.get(key, False):
        _LOGGER.warning(message)
        _WARNED[key] = True


def _as_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    return np.clip(gray, 0, 255).astype(np.uint8)


def _niqe_opencv(gray: np.ndarray) -> float:
    score = cv2.quality.QualityNIQE_compute(gray)
    if isinstance(score, (tuple, list, np.ndarray)):
        score = score[0]
    return float(score)


def _niqe_piq(gray: np.ndarray) -> float:
    import torch
    import piq

    if not hasattr(piq, "niqe"):
        raise AttributeError("piq.niqe not available in this version")

    x = gray.astype(np.float32) / 255.0
    x = x[None, None, :, :]
    with torch.no_grad():
        score = piq.niqe(torch.from_numpy(x), data_range=1.0)
    return float(score.item())


def _niqe_pyiqa(gray: np.ndarray) -> float:
    import torch
    import pyiqa

    global _PYIQA_NIQE
    if _PYIQA_NIQE is None:
        cache_dir = Path.cwd() / ".torch_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))
        _PYIQA_NIQE = pyiqa.create_metric("niqe", device="cpu")
    metric = _PYIQA_NIQE
    x = gray.astype(np.float32) / 255.0
    x = x[None, None, :, :]
    with torch.no_grad():
        score = metric(torch.from_numpy(x))
    return float(score.item())


def compute_niqe(gray: np.ndarray, short_side: int = 512) -> float | None:
    gray = resize_short_side(gray, short=short_side)
    gray_u8 = _as_uint8(gray)

    if supports_opencv_quality() and hasattr(cv2.quality, "QualityNIQE_compute"):
        return _niqe_opencv(gray_u8)

    for fn in (_niqe_pyiqa, _niqe_piq):
        try:
            return fn(gray_u8)
        except Exception:
            continue

    _warn_once(
        "niqe",
        "NIQE unavailable (OpenCV quality missing and no fallback installed).",
    )
    return None


def _brisque_opencv(gray: np.ndarray, model_path: Path, range_path: Path) -> float:
    score = cv2.quality.QualityBRISQUE_compute(gray, str(model_path), str(range_path))
    if isinstance(score, (tuple, list, np.ndarray)):
        score = score[0]
    return float(score)


def _brisque_piq(gray: np.ndarray) -> float:
    import torch
    import piq

    x = gray.astype(np.float32) / 255.0
    x = np.repeat(x[:, :, None], 3, axis=2)
    x = x.transpose(2, 0, 1)[None, :, :, :]
    with torch.no_grad():
        score = piq.brisque(torch.from_numpy(x), data_range=1.0)
    return float(score.item())


def _brisque_imquality(gray: np.ndarray) -> float:
    from PIL import Image
    import imquality.brisque as brisque

    img = Image.fromarray(gray)
    return float(brisque.score(img))


def _brisque_pybrisque(gray: np.ndarray) -> float:
    from brisque import BRISQUE

    scorer = BRISQUE(url=False)
    img = np.repeat(gray[:, :, None], 3, axis=2)
    return float(scorer.score(img))


def compute_brisque(
    gray: np.ndarray,
    model_path: Path,
    range_path: Path,
    short_side: int = 512,
) -> float | None:
    gray = resize_short_side(gray, short=short_side)
    gray_u8 = _as_uint8(gray)

    if supports_opencv_quality() and hasattr(cv2.quality, "QualityBRISQUE_compute"):
        if model_path.exists() and range_path.exists():
            return _brisque_opencv(gray_u8, model_path, range_path)
        _warn_once("brisque", "BRISQUE model assets missing; falling back.")

    for fn in (_brisque_piq, _brisque_imquality, _brisque_pybrisque):
        try:
            return fn(gray_u8)
        except Exception:
            continue

    _warn_once(
        "brisque",
        "BRISQUE unavailable (OpenCV quality missing and no fallback installed).",
    )
    return None
