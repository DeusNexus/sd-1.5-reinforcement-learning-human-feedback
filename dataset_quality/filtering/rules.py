"""Filtering rules for dataset quality metrics."""

from __future__ import annotations

from typing import Any


def _below(val: float | None, threshold: float | None) -> bool:
    return val is not None and threshold is not None and val < threshold


def _above(val: float | None, threshold: float | None) -> bool:
    return val is not None and threshold is not None and val > threshold


def evaluate_metrics(metrics: dict[str, Any], thresholds: dict[str, float | None]) -> tuple[bool, list[str], dict[str, float]]:
    reasons: list[str] = []
    debug: dict[str, float] = {}

    if metrics.get("load_error"):
        return False, ["load_error"], debug

    # Blur / sharpness
    if _below(metrics.get("roi_tenengrad"), thresholds.get("roi_tenengrad_min")):
        reasons.append("roi_blur_tenengrad")
    if _below(metrics.get("roi_lap_var"), thresholds.get("roi_lap_var_min")):
        reasons.append("roi_blur_laplacian")
    if _below(metrics.get("tenengrad"), thresholds.get("global_tenengrad_min")):
        reasons.append("global_blur_tenengrad")
    if _below(metrics.get("lap_var"), thresholds.get("global_lap_var_min")):
        reasons.append("global_blur_laplacian")

    # Compression artifacts
    if _above(metrics.get("jpeg_blockiness"), thresholds.get("jpeg_blockiness_max")):
        reasons.append("jpeg_blockiness")
    if (
        _above(metrics.get("jpeg_blockiness"), thresholds.get("jpeg_blockiness_hi"))
        and _above(metrics.get("niqe"), thresholds.get("niqe_hi"))
    ):
        reasons.append("jpeg_blockiness_niqe_combo")

    # NR-IQA
    if _above(metrics.get("niqe"), thresholds.get("niqe_max")):
        reasons.append("niqe")
    if _above(metrics.get("brisque"), thresholds.get("brisque_max")):
        reasons.append("brisque")

    # Black bars
    if _above(metrics.get("black_bar_score"), thresholds.get("black_bar_score_min")):
        reasons.append("black_bars_score")
    if (
        _above(metrics.get("black_bar_area_ratio"), thresholds.get("black_bar_area_max"))
        and metrics.get("black_bar_solid")
    ):
        reasons.append("black_bars_area")

    keep = len(reasons) == 0
    return keep, reasons, debug
