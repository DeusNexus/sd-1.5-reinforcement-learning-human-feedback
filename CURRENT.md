# Current Changes

This file lists only the latest implemented changes. For full history, see `CHANGES.md`.

## Latest - 25-01-2026
- Dataset quality pipeline now drops the worst 20% by composite badness score (see `dataset_quality/config.py`).
- Dataset quality notebook renamed to `0_dataset_quality_enhance.ipynb`.
- Dataset quality metrics are cached to avoid recomputation when inputs/config match.
- Base LoRA training uses aspect-ratio-preserving resize + crop with bicubic interpolation and reduced color jitter.
- Training now uses `DDPMScheduler` noise and `CosineAnnealingLR` (no restarts) for improved stability.
- Validation loss is computed and tracked; best checkpoints follow validation loss when available.
- Default scratch-run learning rate lowered to 1e-4.
