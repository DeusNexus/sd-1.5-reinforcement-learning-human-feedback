"""Pipeline orchestration for dataset quality metrics and filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from dataset_quality import config
from dataset_quality.filtering.rules import evaluate_metrics
from dataset_quality.io.dataset_loader import load_splits
from dataset_quality.io.dataset_writer import write_dataset
from dataset_quality.io.reports import write_reports
from dataset_quality.metrics import (
    artifact_score,
    black_bar_score,
    compute_brisque,
    compute_niqe,
    compute_roi_sharpness,
    compute_sharpness,
    detect_faces,
    face_area_ratio,
    fallback_roi,
    jpeg_blockiness,
    resize_short_side,
    select_largest_face,
)
from dataset_quality.metrics.black_bars import find_black_bars

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(items, **_kwargs):
        return items


_LOGGER = logging.getLogger(__name__)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def _read_image(image_path: Path) -> tuple[Any | None, Any | None]:
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def _metric_signature(cfg: dict[str, Any], metric_cols: list[str]) -> str:
    payload = {
        "metric_config": cfg,
        "metric_cols": metric_cols,
    }
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_image_metrics(image_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    img, gray = _read_image(image_path)
    if img is None or gray is None:
        return {"load_error": True}

    metrics: dict[str, Any] = {"load_error": False}

    # Sharpness (global + ROI)
    sharpness_cfg = cfg["SHARPNESS"]
    metrics.update(compute_sharpness(gray, short_side=sharpness_cfg["short_side"]))

    face_cfg = cfg["FACE"]
    faces = detect_faces(
        gray,
        cascade_path=face_cfg["cascade_path"],
        scale_factor=face_cfg["scale_factor"],
        min_neighbors=face_cfg["min_neighbors"],
        min_size=face_cfg["min_size"],
    )
    face_bbox = select_largest_face(faces)
    face_found = face_bbox is not None
    if not face_found:
        face_bbox = fallback_roi(gray, ratio=face_cfg["fallback_ratio"])
    metrics["face_found"] = face_found
    metrics["face_area_ratio"] = face_area_ratio(face_bbox if face_found else None, gray.shape)
    metrics.update(
        compute_roi_sharpness(
            gray,
            roi_bbox=face_bbox,
            roi_size=sharpness_cfg["roi_size"],
        )
    )

    # JPEG blockiness
    block_cfg = cfg["BLOCKINESS"]
    gray_block = resize_short_side(gray, short=block_cfg["short_side"])
    blockiness = jpeg_blockiness(gray_block, block_size=block_cfg["block_size"])
    metrics["jpeg_blockiness"] = blockiness
    metrics["artifact_score"] = artifact_score(
        blockiness, metrics.get("tenengrad", 0.0), eps=block_cfg["eps"]
    )

    # NR-IQA
    iqa_cfg = cfg["NR_IQA"]
    metrics["niqe"] = compute_niqe(gray, short_side=iqa_cfg["short_side"])
    metrics["brisque"] = compute_brisque(
        gray,
        model_path=iqa_cfg["brisque_model_path"],
        range_path=iqa_cfg["brisque_range_path"],
        short_side=iqa_cfg["short_side"],
    )

    # Black bars
    bb_cfg = cfg["BLACK_BARS"]
    candidates = find_black_bars(
        gray,
        thr=bb_cfg["thr"],
        min_area_ratio=bb_cfg["min_area_ratio"],
        solid_std_max=bb_cfg["solid_std_max"],
        solid_grad_max=bb_cfg["solid_grad_max"],
        uniform_frac_min=bb_cfg["uniform_frac_min"],
        min_span_ratio=bb_cfg["min_span_ratio"],
        max_thickness_ratio=bb_cfg["max_thickness_ratio"],
    )
    if candidates:
        scored = [
            (black_bar_score(c, min_area_ratio=bb_cfg["min_area_ratio"]), c)
            for c in candidates
        ]
        best_score, best = max(scored, key=lambda x: x[0])
        metrics["black_bar_score"] = best_score
        metrics["black_bar_area_ratio"] = best.area_ratio
        metrics["black_bar_span_ratio"] = max(best.span_w_ratio, best.span_h_ratio)
        metrics["black_bar_solid"] = bool(best.solid)
        metrics["black_bar_count"] = len(candidates)
    else:
        metrics["black_bar_score"] = 0.0
        metrics["black_bar_area_ratio"] = 0.0
        metrics["black_bar_span_ratio"] = 0.0
        metrics["black_bar_solid"] = False
        metrics["black_bar_count"] = 0

    return metrics


def _compute_badness(
    df: pd.DataFrame,
    higher_is_worse: list[str],
    lower_is_worse: list[str],
) -> pd.Series:
    parts = []
    for metric in higher_is_worse:
        if metric in df.columns:
            parts.append(df[metric].rank(pct=True, ascending=True, na_option="bottom"))
    for metric in lower_is_worse:
        if metric in df.columns:
            parts.append(df[metric].rank(pct=True, ascending=False, na_option="bottom"))
    if not parts:
        return pd.Series([0.0] * len(df), index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def run_pipeline(
    dataset_root: Path,
    output_root: Path,
    report_dir: Path | None = None,
    link_mode: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    report_dir = report_dir or (output_root / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message="IProgress not found.*")
    cfg = {
        "SHARPNESS": config.SHARPNESS,
        "FACE": config.FACE,
        "BLOCKINESS": config.BLOCKINESS,
        "NR_IQA": config.NR_IQA,
        "BLACK_BARS": config.BLACK_BARS,
        "THRESH": config.THRESH,
        "PIPELINE": config.PIPELINE,
    }
    if cfg["NR_IQA"].get("warn_once", True):
        if not (cfg["NR_IQA"]["brisque_model_path"].exists() and cfg["NR_IQA"]["brisque_range_path"].exists()):
            _LOGGER.warning("BRISQUE model assets missing; falling back.")

    full_df = load_splits(dataset_root)
    if limit:
        full_df = full_df.head(limit).copy()

    metric_cols = [
        "load_error",
        "lap_var",
        "tenengrad",
        "roi_lap_var",
        "roi_tenengrad",
        "face_found",
        "face_area_ratio",
        "jpeg_blockiness",
        "artifact_score",
        "niqe",
        "brisque",
        "black_bar_score",
        "black_bar_area_ratio",
        "black_bar_span_ratio",
        "black_bar_solid",
        "black_bar_count",
    ]
    base_cols = ["imageId", "split", "age", "gender", "ethnicity", "emotion", "image_path"]
    required_cols = set(base_cols + metric_cols)

    cache_cfg = cfg["PIPELINE"]
    cache_enabled = cache_cfg.get("metrics_cache_enabled", True)
    cache_csv = report_dir / cache_cfg.get("metrics_cache_name", "quality_metrics_cache.csv")
    cache_meta = report_dir / cache_cfg.get("metrics_cache_meta", "quality_metrics_cache.json")
    dataset_paths = set(full_df["image_path"].astype(str))

    cache_signature = _metric_signature(
        {
            "SHARPNESS": cfg["SHARPNESS"],
            "FACE": cfg["FACE"],
            "BLOCKINESS": cfg["BLOCKINESS"],
            "NR_IQA": cfg["NR_IQA"],
            "BLACK_BARS": cfg["BLACK_BARS"],
        },
        metric_cols,
    )

    quality_df: pd.DataFrame | None = None
    if cache_enabled and cache_csv.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if meta.get("signature") == cache_signature:
                cached_df = pd.read_csv(cache_csv)
                if required_cols.issubset(cached_df.columns) and set(cached_df["image_path"].astype(str)) == dataset_paths:
                    quality_df = cached_df
                    _LOGGER.info("Using cached metrics at %s", cache_csv)
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning("Failed to load metrics cache: %s", exc)

    if quality_df is None:
        rows = []
        for row in tqdm(full_df.itertuples(index=False), total=len(full_df), mininterval=1.0):
            metrics = compute_image_metrics(Path(row.image_path), cfg)
            rows.append(
                {
                    "imageId": int(row.imageId),
                    "split": row.split,
                    "age": int(row.age),
                    "gender": row.gender,
                    "ethnicity": row.ethnicity,
                    "emotion": row.emotion,
                    "image_path": str(row.image_path),
                    **metrics,
                }
            )
        quality_df = pd.DataFrame(rows)

        if cache_enabled:
            quality_df.to_csv(cache_csv, index=False)
            cache_meta.write_text(
                json.dumps(
                    {
                        "signature": cache_signature,
                        "dataset_root": str(dataset_root),
                        "image_count": len(quality_df),
                        "metric_cols": metric_cols,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    drop_reasons = []
    keeps = []
    for row in quality_df.itertuples(index=False):
        keep, reasons, _debug = evaluate_metrics(row._asdict(), cfg["THRESH"])
        drop_reasons.append(reasons)
        keeps.append(keep)

    quality_df["drop_reasons"] = drop_reasons
    quality_df["keep"] = keeps

    drop_worst_frac = cfg["PIPELINE"].get("drop_worst_frac", 0.0)
    if drop_worst_frac and drop_worst_frac > 0:
        quality_df["badness_score"] = _compute_badness(
            quality_df,
            cfg["PIPELINE"].get("drop_worst_metrics_hi", []),
            cfg["PIPELINE"].get("drop_worst_metrics_lo", []),
        )
        cutoff = quality_df["badness_score"].quantile(1 - drop_worst_frac)
        worst_mask = quality_df["badness_score"] > cutoff
        for idx, is_worst in enumerate(worst_mask):
            if is_worst:
                drop_reasons[idx].append("worst_quality_percentile")
        quality_df["keep"] = quality_df["keep"] & ~worst_mask

    quality_df["drop_reasons_str"] = quality_df["drop_reasons"].apply(
        lambda r: ";".join(r) if r else ""
    )

    write_reports(
        quality_df,
        report_dir=report_dir,
        worst_metrics=["roi_tenengrad", "roi_lap_var", "jpeg_blockiness", "niqe", "black_bar_score"],
    )

    kept_df = quality_df[quality_df["keep"]].copy()
    write_dataset(
        kept_df,
        output_root=output_root,
        link_mode=link_mode or config.PIPELINE["link_mode"],
    )

    return quality_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset quality pipeline.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        report_dir=args.report_dir,
        link_mode=args.link_mode,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
