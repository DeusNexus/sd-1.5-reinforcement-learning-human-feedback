# Current Changes

This file lists only the latest implemented changes. For full history, see `CHANGES.md`.

## Latest - 29-01-2026
- Tightened dataset quality hard thresholds (NIQE/BRISQUE/blur/black bars/JPEG blockiness) in `dataset_quality/config.py`.
- Worst-percentile pruning is stratified by 5-year age bins with a minimum keep of 200 per bin.
- Training notebooks now cap per-age-bin samples using `quality_report.csv` to keep the best images.
- Updated dataset cleaning docs and latest stats in `README.md`, `REPORT.md`, and `datasets/README.md`.
