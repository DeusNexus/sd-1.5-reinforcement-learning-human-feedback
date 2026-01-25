# Change Log

This file records implemented changes in chronological order. New entries go at the top.

## Entry 019 (latest) - 25-01-2026
- Dataset quality pipeline now drops the worst 20% by composite badness score (configurable in `dataset_quality/config.py`).
- Renamed the dataset quality notebook to `0_dataset_quality_enhance.ipynb`.
- Added metrics caching to reuse computed quality scores when inputs/config match.

## Entry 018 - 25-01-2026
- Base LoRA training now uses aspect-ratio-preserving resize + crop with bicubic interpolation and reduced color jitter.
- Switched training noise scheduler to `DDPMScheduler` and replaced cosine restarts with `CosineAnnealingLR` for stability.
- Added validation loss computation and tracking; checkpoints now follow validation loss when available.
- Updated default scratch-run learning rate to 1e-4 for smoother convergence.

## Entry 017 - 25-01-2025
- Documented the cleaned dataset at `datasets/appa-real-dataset_v2_improved` as the base LoRA training source.
- Updated dataset quality notebook references to `0_dataset_quality_enhance.ipynb` and `0_dataset_quality_review.ipynb`.
- Removed stale references to the black-bar review notebook.

## Entry 016 - 24-01-2025
- Replaced `5_dataset_improvement.ipynb` with the modular dataset quality pipeline (`dataset_quality/`) and notebooks `0_dataset_quality_enhance.ipynb` + `0_dataset_quality_review.ipynb`.
- Removed `5_dataset_improvement.ipynb` from the repo.

## Entry 015 - 23-01-2025
- Relaxed face detection defaults in `5_dataset_improvement.ipynb` (scale factor, neighbors, and min size) to reduce false `no_face`/`small_face` flags.
- Loosened blur/noise and face-area defaults in `5_dataset_improvement.ipynb` to keep more valid images while still filtering blur/noise.
- Extended dropped-image overlays in `5_dataset_improvement.ipynb` to include values for compression/noise metrics (blockiness, chroma noise, face blur).

## Entry 014 - 23-01-2025
- Updated dropped-image labels in `5_dataset_improvement.ipynb` to show metric values (e.g., `blur:25.0`, `black_bars:0.05`).

## Entry 013 - 23-01-2025
- Fixed the dropped-image metrics inspection cell formatting in `5_dataset_improvement.ipynb` (newline prints and quote escaping).

## Entry 012 - 23-01-2025
- Added a dropped-image metrics inspection cell in `5_dataset_improvement.ipynb` to show per-image metrics and per-reason summaries for threshold tuning.

## Entry 011 - 23-01-2025
- Updated black-bar detection to require a 25x25 black patch that touches the border, and lowered the black pixel threshold for stricter “true black” detection.

## Entry 010 - 23-01-2025
- Fixed the histogram grid in `5_dataset_improvement.ipynb` to handle 7 metrics without index errors.

## Entry 009 - 23-01-2025
- Relaxed face filtering (optional `require_face`, lower `min_face_area_ratio`) and loosened noise threshold in `5_dataset_improvement.ipynb`; added black-bar detection via black pixel ratio + connected-component area.

## Entry 008 - 23-01-2025
- Fixed a string literal bug in `5_dataset_improvement.ipynb` that caused a `SyntaxError` in `_parse_reasons`.

## Entry 007 - 23-01-2025
- Improved dropped-image label rendering in `5_dataset_improvement.ipynb` (parses stringified reasons, shortens labels, larger font, and more robust text sizing).

## Entry 006 - 23-01-2025
- Updated the dropped-image sample grid in `5_dataset_improvement.ipynb` to overlay the drop reason on each image.

## Entry 005 - 23-01-2025
- Tightened quality thresholds in `5_dataset_improvement.ipynb` to focus on blur/noise filtering (min blur 60, max noise 3.5) and relaxed face area ratio to 0.25.

## Entry 004 - 23-01-2025
- Synced Colab notebooks to use non-Colab notebooks as source of truth.
- `1_diffusion_rl_base_colab_v2.ipynb` now matches `1_diffusion_rl_base.ipynb` logic; Colab-only setup cells remain; GPU cleanup cell kept and placed after imports.
- `3_1_diffusion_sample_rlhf_colab.ipynb` now matches `3_1_diffusion_sample_rlhf.ipynb` logic; Colab-only setup and optional zip/download remain.
- Added dataset quality improvement plan in `DATASET_IMPROVEMENTS.md` covering filtering, alignment, label QA, and SD1.5 vs SDXL upgrade strategy.
- Added dataset improvement notebook `5_dataset_improvement.ipynb` to compute quality metrics, filter images, rebuild `datasets/appa-real-dataset_v2_improved`, and visualize dropped vs kept samples.

## Entry 003 - 23-01-2025
- RLHF generation pipeline added in `3_1_diffusion_sample_rlhf.ipynb` and `3_1_diffusion_sample_rlhf_colab.ipynb` to generate A/B samples, save latent tensors, and append to `ratings_interface/rlhf_generation_data/generation_manifest.jsonl`.
- Prompt hashing + resume-safe generation added (skips existing samples using the manifest).
- Human rating UI implemented in `4_hf_rating.ipynb` using Gradio, with prompt display and left/right/skip actions, writing to `ratings_interface/ratings_real.jsonl`.

## Entry 002 - 23-01-2025
- LoRA checkpoint saving updated to store a small adapter file (`adapter_model.safetensors`) plus a full UNet checkpoint for resume in `1_diffusion_rl_base.ipynb` and `1_diffusion_rl_base_colab_v2.ipynb`.
- Training history and sample images saved per run in `lora_training_runs/<run_id>/`.

## Entry 001 - 23-01-2025
- Base supervised LoRA fine-tuning implemented in `1_diffusion_rl_base.ipynb` and `1_diffusion_rl_base_colab_v2.ipynb`.
- Prompt generation helper added in `2_generate_prompts.ipynb`.
- Single-image inference notebook added in `3_0_generate_sample_image.ipynb`.
