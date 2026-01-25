"""Black bar detection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BarCandidate:
    bbox: tuple[int, int, int, int]
    area_ratio: float
    border_touch: bool
    span_w_ratio: float
    span_h_ratio: float
    thickness_w_ratio: float
    thickness_h_ratio: float
    bar_like: bool
    solid: bool
    uniform_frac: float
    grad_mean: float
    std: float


def _solid_stats(
    gray: np.ndarray,
    component_mask: np.ndarray,
    thr: int,
) -> tuple[float, float, float]:
    region = gray[component_mask]
    if region.size == 0:
        return 0.0, 0.0, 0.0
    std = float(region.std())
    uniform_frac = float((region <= (thr + 2)).mean())
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(grad[component_mask].mean())
    return std, grad_mean, uniform_frac


def is_solid_black_region(
    gray: np.ndarray,
    component_mask: np.ndarray,
    thr: int = 8,
    solid_std_max: float = 8.0,
    solid_grad_max: float = 5.0,
    uniform_frac_min: float = 0.9,
) -> bool:
    std, grad_mean, uniform_frac = _solid_stats(gray, component_mask, thr)
    return (
        std <= solid_std_max
        and grad_mean <= solid_grad_max
        and uniform_frac >= uniform_frac_min
    )


def find_black_bars(
    gray: np.ndarray,
    thr: int = 8,
    min_area_ratio: float = 0.03,
    solid_std_max: float = 8.0,
    solid_grad_max: float = 5.0,
    uniform_frac_min: float = 0.9,
    min_span_ratio: float = 0.9,
    max_thickness_ratio: float = 0.25,
) -> list[BarCandidate]:
    mask = gray <= thr
    if not mask.any():
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    h, w = gray.shape[:2]
    image_area = float(h * w)
    candidates = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = float(area) / image_area
        if area_ratio < min_area_ratio:
            continue
        border_touch = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        if not border_touch:
            continue

        span_w_ratio = float(bw) / float(w)
        span_h_ratio = float(bh) / float(h)
        thickness_w_ratio = float(bw) / float(w)
        thickness_h_ratio = float(bh) / float(h)
        bar_like = (
            (span_w_ratio >= min_span_ratio and thickness_h_ratio <= max_thickness_ratio)
            or (span_h_ratio >= min_span_ratio and thickness_w_ratio <= max_thickness_ratio)
        )

        component_mask = labels == label
        std, grad_mean, uniform_frac = _solid_stats(gray, component_mask, thr)
        solid = (
            std <= solid_std_max
            and grad_mean <= solid_grad_max
            and uniform_frac >= uniform_frac_min
        )

        candidates.append(
            BarCandidate(
                bbox=(x, y, bw, bh),
                area_ratio=area_ratio,
                border_touch=border_touch,
                span_w_ratio=span_w_ratio,
                span_h_ratio=span_h_ratio,
                thickness_w_ratio=thickness_w_ratio,
                thickness_h_ratio=thickness_h_ratio,
                bar_like=bar_like,
                solid=solid,
                uniform_frac=uniform_frac,
                grad_mean=grad_mean,
                std=std,
            )
        )
    return candidates


def black_bar_score(
    candidate: BarCandidate,
    min_area_ratio: float = 0.03,
) -> float:
    if not candidate.border_touch:
        return 0.0
    score = 0.0
    score += 0.2
    score += min(1.0, candidate.area_ratio / max(min_area_ratio, 1e-6)) * 0.3
    score += candidate.uniform_frac * 0.3
    score += max(candidate.span_w_ratio, candidate.span_h_ratio) * 0.2
    if not candidate.bar_like:
        score *= 0.5
    if not candidate.solid:
        score *= 0.2
    return float(min(1.0, score))
