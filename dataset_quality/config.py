"""Default configuration for dataset quality pipeline."""

from __future__ import annotations

from pathlib import Path

SPLIT_DIRS = {
    "train": "train_data",
    "valid": "valid_data",
    "test": "test_data",
}

SHARPNESS = {
    "short_side": 256,
    "roi_size": (160, 160),
}

FACE = {
    "cascade_path": None,  # If None, uses cv2.data.haarcascades default.
    "scale_factor": 1.05,
    "min_neighbors": 3,
    "min_size": (24, 24),
    "fallback_ratio": 0.6,
}

BLOCKINESS = {
    "block_size": 8,
    "short_side": 256,
    "eps": 1e-6,
}

NR_IQA = {
    "short_side": 512,
    "brisque_model_path": Path("models/brisque_model_live.yml"),
    "brisque_range_path": Path("models/brisque_range_live.yml"),
    "warn_once": True,
}

BLACK_BARS = {
    "thr": 8,
    "min_area_ratio": 0.03,
    "solid_std_max": 8.0,
    "solid_grad_max": 5.0,
    "uniform_frac_min": 0.9,
    "min_span_ratio": 0.9,
    "max_thickness_ratio": 0.25,
}

THRESH = {
    # Blur / sharpness
    "roi_tenengrad_min": 20.0,
    "roi_lap_var_min": 15.0,
    "global_tenengrad_min": None,
    "global_lap_var_min": None,
    # Compression artifacts
    "jpeg_blockiness_max": 2.0,
    "jpeg_blockiness_hi": 15.0,
    "niqe_hi": 11.7,
    # NR-IQA
    "niqe_max": 10.6,
    "brisque_max": 73.2,
    # Black bars
    "black_bar_score_min": 0.5,
    "black_bar_area_max": 0.20,
}

PIPELINE = {
    "link_mode": "copy",  # copy, hardlink, symlink
    "random_seed": 42,
    "sample_grid_size": 16,
}
