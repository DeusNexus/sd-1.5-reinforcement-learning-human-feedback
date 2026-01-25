# Diffusion RLHF for Attribute-Conditioned Portraits

This repo builds a Stable Diffusion LoRA that follows prompts describing **age, gender, ethnicity, and emotion**, then improves results with **human preference data** (RLHF/DPO). The project focuses on:
- Curating a clean, labeled face dataset.
- Training a base LoRA on text-conditioned labels.
- Generating A/B samples for human rating.
- (Planned) Preference tuning with a latent-space DPO loss.

## Quick start
From repo root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install opencv-contrib-python numpy pandas pillow matplotlib tqdm torch pyiqa piq jupyter
```

Run the dataset quality pipeline:
```bash
python -m dataset_quality.pipeline \
  --dataset-root datasets/appa-real-dataset_v2 \
  --output-root datasets/appa-real-dataset_v2_improved
```

Open the review notebook:
```bash
jupyter lab
```
Then run:
- `notebooks/dataset_quality.ipynb`
- `notebooks/dataset_quality_review.ipynb`
- `notebooks/black_bar_review.ipynb`

## Repository map (what to look at)

### Dataset quality (cleaning + analysis)
- `dataset_quality/` — modular pipeline (metrics, rules, reporting).
- `notebooks/dataset_quality.ipynb` — runs the pipeline.
- `notebooks/dataset_quality_review.ipynb` — visual review, metric distributions, label distributions.
- `notebooks/black_bar_review.ipynb` — focused inspection of black-bar cases.

### Base LoRA training
- `1_diffusion_rl_base.ipynb` — local training notebook.
- `1_diffusion_rl_base_colab_v2.ipynb` — Colab version.

### RLHF data generation + rating
- `2_generate_prompts.ipynb` — prompt grid builder.
- `3_0_generate_sample_image.ipynb` — single prompt inference.
- `3_1_diffusion_sample_rlhf.ipynb` — A/B sample generation + latents.
- `4_hf_rating.ipynb` — human rating UI.
- `ratings_interface/` — generated samples + ratings JSONL.

## Dataset
We start from:
- `datasets/appa-real-dataset_v2` (original labeled dataset)

We produce a cleaned dataset at:
- `datasets/appa-real-dataset_v2_improved`

The improved dataset is rebuilt by filtering low-quality images based on:
- Sharpness (Laplacian variance + Tenengrad)
- Compression artifacts (JPEG blockiness)
- NR-IQA (NIQE + BRISQUE)
- Black bars (solid border bars)

## Quality distributions (before vs kept)
These plots are generated from `datasets/appa-real-dataset_v2_improved/reports/quality_report.csv` and saved to `docs/figures/`.

![Metric distributions](docs/figures/metric_distributions.png)
![Label distributions](docs/figures/label_distributions.png)

## Outputs
- Cleaned dataset: `datasets/appa-real-dataset_v2_improved/`
- Reports: `datasets/appa-real-dataset_v2_improved/reports/`
  - `quality_report.csv`
  - `kept_images.csv`
  - `dropped_images.csv`
  - `drop_reason_counts.csv`

## Notes on NIQE/BRISQUE
- NIQE is computed with `pyiqa` (CPU).
- BRISQUE uses `opencv-contrib-python` if model files exist, otherwise falls back to `piq`.
- If you want OpenCV BRISQUE, place model files at:
  - `models/brisque_model_live.yml`
  - `models/brisque_range_live.yml`
