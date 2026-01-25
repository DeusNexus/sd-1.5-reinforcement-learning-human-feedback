# Dataset Quality Improvement Plan

This document outlines how to improve the current training dataset so the LoRA learns cleaner facial structure and fewer artifacts. It also explains how dataset quality impacts the decision to upgrade from SD1.5 to SDXL.

## Why this matters
The current dataset includes blurry, noisy, and low-quality images. When the LoRA learns from these, it will reproduce the same issues during generation (odd mouth/teeth/eye shapes, unstable facial geometry, and low-fidelity textures). Improving data quality directly improves prompt fidelity and visual realism.

## Goals
- Remove low-quality images that inject distortion into the model.
- Improve consistency of facial geometry and alignment.
- Reduce label noise for age, gender, ethnicity, and emotion.
- Produce a clean, balanced, and higher-quality training set for LoRA and RLHF.

## Current pipeline entrypoints
- Script: `python -m dataset_quality.pipeline --dataset-root datasets/appa-real-dataset_v2 --output-root datasets/appa-real-dataset_v2_improved`
- Notebook: `0_dataset_quality_enhance.ipynb`
- Review notebook: `0_dataset_quality_review.ipynb`
- Config: `dataset_quality/config.py`
- Cleaned dataset: `datasets/appa-real-dataset_v2_improved` (same layout as `datasets/appa-real-dataset_v2`)
- Pipeline drops the worst 20% by composite badness score (see `dataset_quality/config.py`).
- Quality metrics are cached under the output `reports/` folder when inputs/config match.

## NIQE/BRISQUE dependencies
- Preferred: OpenCV quality module via `opencv-contrib-python` (uses `cv2.quality`).
- Fallbacks supported in code:
  - NIQE: `pyiqa`
  - BRISQUE: `piq`, `imquality`, or `brisque`

## Phase 1: Audit and quantify data quality
Create a quality audit script to produce a CSV report per image with:
- Blur score (Laplacian variance or similar).
- Face detection confidence and bounding box size.
- Brightness and contrast range (under/overexposure).
- Estimated noise level (simple noise estimator).
- Optional: face alignment/landmark confidence.

Use histograms to decide thresholds instead of hard-coded cutoffs at first. The aim is to find the lowest-quality tail, then review a sample before filtering.

## Phase 2: Filter the worst-quality images
Remove or quarantine images that fail the most basic quality checks:
- No face detected or low detection confidence.
- Very low blur score (extreme blur).
- Severe under/overexposure.
- Very small face area (face occupies too little of the image).

Keep a separate "rejected" folder so nothing is lost permanently.

## Phase 3: Face alignment and crop normalization
To reduce odd facial geometry:
- Detect face landmarks.
- Align and crop to a consistent framing (eyes roughly level, centered face).
- Resize to the model target size (512 for SD1.5).

This reduces pose variance and allows the LoRA to focus on attributes rather than camera pose.

## Phase 4: Label quality checks
Label noise can cause contradictions during training.
- Run a lightweight classifier (age/gender/emotion) to flag likely mismatches.
- Review or discard images where prediction conflicts with label.
- Track label distribution to keep classes reasonably balanced.

## Phase 5: Rebuild the training split
After filtering, rebuild train/valid/test splits to preserve class balance. If some classes become too small, consider:
- Collecting more images.
- Reducing prompt granularity for those attributes.

## Phase 6: Retrain LoRA on cleaned dataset
Once the cleaned dataset is ready:
- Retrain LoRA from scratch.
- Use `datasets/appa-real-dataset_v2_improved` as the training root in `1_diffusion_rl_base.ipynb` and `1_diffusion_rl_base_colab_v2.ipynb`.
- Compare generation quality against the current LoRA.
- Only then resume RLHF data generation and human labeling.

## SD1.5 vs SDXL: Upgrade strategy
### Current reality
Your dataset images are low-res (224x224) and contain quality issues. Training SDXL (which is tuned for 1024x1024) with noisy low-res data will likely amplify artifacts, not reduce them.

### Recommended strategy
1. **Stay on SD1.5 for now** and focus on dataset cleanup + retraining.
2. **Upgrade to SDXL only after**:
   - A clean, higher-quality dataset is ready.
   - Faces are aligned and consistently framed.
   - You can reliably produce 768 or 1024 crops without heavy blur.

### If SDXL is required sooner
If you must test SDXL now:
- Use only the cleanest subset.
- Apply face alignment and conservative upscaling.
- Expect limited gains until data quality improves.

## Impact on RLHF
RLHF does not replace good data. If the base LoRA is trained on noisy faces, human preference tuning will be forced to choose between bad outputs. The most cost-effective move is to fix dataset quality **before** scaling human labeling.

## Summary checklist
- [ ] Audit dataset and build quality report.
- [ ] Filter low-quality / no-face / tiny-face images.
- [ ] Align + crop faces to stable framing.
- [ ] Validate labels and rebalance splits.
- [ ] Retrain LoRA on cleaned data.
- [ ] Regenerate RLHF samples and collect new ratings.
- [ ] Only then consider SDXL migration.
