# Project Report: Diffusion RLHF for Attribute-Conditioned Portrait Generation

## Plain-language overview (what this project is trying to do)
You are building a system that can generate realistic portraits of people from text descriptions, where the description includes specific attributes like age, gender, ethnicity, and emotion. The model should understand prompts like:

```
A 25 years old asian female smiling happily
```

The project uses two big stages:

1. **Base training (supervised learning)**: Teach a Stable Diffusion model to map these prompts to images using a labeled dataset of real faces.
2. **Human preference tuning (RLHF)**: Generate multiple candidate images per prompt, have a human choose the best one, and use those choices to refine the model further using preference-based training (DPO).

The long-term goal is to create a model that not only produces realistic faces, but also follows the prompt more accurately and consistently for age, ethnicity, gender, and emotion.

## High-level pipeline in simple terms
1. **Prepare labeled data**: You have a dataset of real images where each image is tagged with age, gender, ethnicity, and emotion.
2. **Turn labels into text prompts**: A prompt is built from the labels (for example, "A 45 years old caucasian male with a neutral expression").
3. **Train a base model with LoRA**: You fine-tune Stable Diffusion using LoRA so it learns to generate images that match those prompts.
4. **Generate candidate images for RLHF**: For each prompt, generate multiple images (A/B pairs) and save:
   - the image
   - the latent noise used to create it
   - the seed and generation settings
5. **Collect human preferences**: A human rater picks which image is better for the prompt (or skips if both are bad).
6. **(Planned) Train with DPO**: Use the preference data and saved latents to fine-tune the LoRA weights without decoding full images again.

## Current status (based on what exists in the repo)
### Base training is done and saved
- **Dataset is present** in `datasets/appa-real-dataset_v2_improved` (same layout as `datasets/appa-real-dataset_v2`).
- **Base LoRA should be retrained** using `datasets/appa-real-dataset_v2_improved`.
- **LoRA training notebooks exist** and have produced training runs:
  - `lora_training_runs/519037_run_20250903-160027` (5 epochs)
  - `lora_training_runs/54a57c_run_20250904-075419` (24 epochs)
- **Checkpoints are saved**:
  - Small LoRA adapter file: `adapter_model.safetensors` (about 51 MB)
  - Full U-Net checkpoint for resume: `best_full_unet_checkpoint.pth` (about 1.77 GB)

### RLHF data generation is active
- There is a **prompt list** already generated: `prompts_age_inc_5.txt` (432 prompts).
- There is a **generation manifest** with saved latents:
  - `ratings_interface/rlhf_generation_data/generation_manifest.jsonl`
  - Current count: 400 prompt hashes, 800 samples (A/B pairs)
- **Important:** the current RLHF generator saves the **initial noise latents** (the random starting tensor), not the final denoised latent (x0). This is a valid choice for latent-space DPO, but it must be the intended representation before scaling human labeling.

### Human rating is underway
- `ratings_interface/ratings_real.jsonl` contains 108 ratings:
  - 106 marked as PREFERRED
  - 2 marked as SKIPPED_BAD_FIT
- There is also a smaller test set in `ratings_interface/test_images` with its own manifest and ratings.

### Preference training (DPO) is not implemented yet
There is no notebook or script that trains LoRA with the preference data yet. The design is described in the doc, but code has not been added for the DPO training loop.

## Data and assets (what is already here)
### Dataset
Located in `datasets/appa-real-dataset_v2_improved`:
- `labels_metadata_train.csv` (4,065 rows)
- `labels_metadata_valid.csv` (1,482 rows)
- `labels_metadata_test.csv` (1,978 rows)

Each CSV has the same columns:
```
imageId, age, gender, ethnicity, emotion
```

The dataset structure is documented in `datasets/README.md`.

### Training outputs
Training runs live in `lora_training_runs`:
- Each run includes:
  - `training_history.json` (loss per epoch, sample prompts, checkpoint paths)
  - `lora_samples/` (sample images per epoch)
  - `lora_checkpoints/` with:
    - `best_lora_adapter/adapter_model.safetensors`
    - `best_full_unet_checkpoint.pth`

### RLHF generation outputs
Stored under `ratings_interface/rlhf_generation_data`:
- Subfolders per prompt hash (10-char hash)
- Each folder contains:
  - `*_A.png`, `*_B.png`
  - `*_A_latent.pt`, `*_B_latent.pt`
- A global manifest: `generation_manifest.jsonl`

### Ratings outputs
Human preferences are saved in JSONL:
- `ratings_interface/ratings_real.jsonl`
- `ratings_interface/test_images/ratings.jsonl`

## Notes from the RLHF In-Depth Summary doc
The document `docs/RLHF In-Depth Summary (6 Oct).docx` describes the overall plan and matches the code direction. Key points:
- The base LoRA fine-tuning is considered complete.
- RLHF will use **pairwise ranking** (A vs B) rather than good/bad.
- **DPO** is chosen instead of PPO to avoid training a separate reward model.
- The plan emphasizes **latent-space DPO**, using saved latents for efficiency.
- It recommends an optional **contrastive loss** over UNet hidden states to improve semantic alignment (age, emotion, ethnicity).

This plan is consistent with what is built so far, except the DPO training loop is still missing in code.

## Notebook-by-notebook breakdown (dedicated section per notebook)

### `1_diffusion_rl_base.ipynb`
**Purpose:** Core local notebook for supervised LoRA training.

**What it does:**
- Loads the dataset CSVs and image folders.
- Builds prompts from labels using the template:
  ```
  A {age} years old {ethnicity} {gender} {emotion phrase}
  ```
- Preprocesses images:
  - Resize to 512x512
  - Random horizontal flip
  - Color jitter
  - Normalize to [-1, 1]
- Loads Stable Diffusion v1.5 and injects LoRA into UNet attention layers (`to_q`, `to_v`).
- Freezes VAE and text encoder; only LoRA parameters are trainable.
- Uses mixed precision: base model in FP16, LoRA in FP32.
- Trains with:
  - VAE encoding
  - Random noise injection via the scheduler
  - MSE loss between predicted noise and true noise
- Saves:
  - Small LoRA adapter in safetensors format
  - Full UNet checkpoint for resume
  - Sample images per epoch
  - Training history JSON

**Outputs:**
- `lora_training_runs/<run_id>/lora_checkpoints`
- `lora_training_runs/<run_id>/lora_samples`
- `lora_training_runs/<run_id>/training_history.json`

### `1_diffusion_rl_base_colab_v2.ipynb`
**Purpose:** Colab-ready version of the base training notebook.

**What it does:**
- Same training logic as `1_diffusion_rl_base.ipynb`.
- Adds Colab-specific steps:
  - `pip install` of dependencies
  - Google Drive mount
  - Path configuration for Drive folders

**Outputs:**
- Same output structure as the local notebook, but intended for Drive.
 
**Sync note:** This Colab notebook mirrors `1_diffusion_rl_base.ipynb` for all training logic; only Colab setup and a GPU cleanup cell differ.

### `2_generate_prompts.ipynb`
**Purpose:** Generates prompt lists for RLHF generation.

**What it does:**
- Defines `generate_prompts(...)` with parameters:
  - `age_increment`
  - `use_gender`
  - `use_ethnicity`
  - `use_emotion`
- Builds all combinations and writes them to a text file.

**Outputs:**
- Example output file: `prompts_age_inc_5.txt`

### `3_0_generate_sample_image.ipynb`
**Purpose:** Simple inference notebook for single prompt generation.

**What it does:**
- Loads a trained LoRA adapter and merges it into Stable Diffusion.
- Builds one prompt from the dataset.
- Runs inference with a fixed seed for reproducibility.

**Outputs:**
- A single generated image shown in the notebook.

### `3_1_diffusion_sample_rlhf.ipynb`
**Purpose:** RLHF dataset generation for A/B image pairs with latents.

**What it does:**
- Loads prompts from a text file and hashes each prompt.
- Loads LoRA weights for inference.
- Generates two samples (A/B) per prompt.
- Saves:
  - Image files (`*_A.png`, `*_B.png`)
  - Latent tensors (`*_A_latent.pt`, `*_B_latent.pt`)
  - Metadata for each sample in `generation_manifest.jsonl`
- Supports:
  - Skipping already-generated samples
  - Running a subset of prompts by index range

**Outputs:**
- `ratings_interface/rlhf_generation_data/`
- `ratings_interface/rlhf_generation_data/generation_manifest.jsonl`

### `3_1_diffusion_sample_rlhf_colab.ipynb`
**Purpose:** Colab version of the RLHF generation notebook.

**What it does:**
- Same generation logic as the local version.
- Adds Colab setup:
  - Install dependencies
  - Clone repo
  - Mount Drive
  - Option to zip and download generated data

**Outputs:**
- Same output structure, but intended to be stored in Drive.
 
**Sync note:** This Colab notebook mirrors `3_1_diffusion_sample_rlhf.ipynb` for all generation logic; only Colab setup and an optional zip/download step differ.

### `4_hf_rating.ipynb`
**Purpose:** Human rating interface for preference collection.

**What it does:**
- Scans `ratings_interface/rlhf_generation_data` for A/B pairs.
- Loads the prompt text from the manifest file.
- Presents images in a Gradio UI.
- Provides buttons:
  - "Left is Better"
  - "Right is Better"
  - "Skip / Bad Pair"
- Saves ratings to JSONL with timestamps.

**Outputs:**
- `ratings_interface/ratings_real.jsonl`

### `dataset_quality/` + review notebooks
**Purpose:** Modular dataset quality pipeline with visual review.

**What it does:**
- Computes sharpness (Laplacian/Tenengrad), blockiness, NIQE/BRISQUE, and black-bar metrics.
- Applies configurable thresholds and records drop reasons.
- Rebuilds a cleaned dataset in `datasets/appa-real-dataset_v2_improved`.
- Produces reports and visual grids of kept vs dropped images.

**Outputs:**
- `datasets/appa-real-dataset_v2_improved/` (when run)
- `datasets/appa-real-dataset_v2_improved/reports/quality_report.csv`
- `datasets/appa-real-dataset_v2_improved/reports/dropped_images.csv`
- `datasets/appa-real-dataset_v2_improved/reports/kept_images.csv`
- Review notebooks: `0_dataset_quality.ipynb`, `0_dataset_quality_review.ipynb`

## How this differs from the old chats
The old chats focused on planning and feasibility. The repo now shows that:
- Base LoRA training **has been implemented and run**, with real checkpoints.
- RLHF data generation **exists and is producing A/B samples** with latents and manifest files.
- A human rating workflow **exists and has produced preference data**.
- The **DPO training stage is still missing** in code (only documented in the docx summary).

So the project has moved from concept to real artifacts, but has not yet completed the preference-training loop.

## Gaps and next steps
1. **Retrain the base LoRA on the improved dataset**:
   - Point `1_diffusion_rl_base.ipynb` and `1_diffusion_rl_base_colab_v2.ipynb` at `datasets/appa-real-dataset_v2_improved`.
   - Regenerate RLHF samples after the new base LoRA is trained.
2. **Implement DPO training**:
   - Build a dataset loader that reads `ratings_real.jsonl` and resolves paths in `generation_manifest.jsonl`.
   - Load the saved latents and compute a preference loss (DPO-style).
   - Update only the LoRA parameters.
3. **Confirm latent representation before scaling labeling**:
   - Current RLHF generation saves **initial noise latents**. If DPO is meant to use x0 latents, update generation now and regenerate before collecting more human ratings.
4. **Unify prompt coverage**:
   - Decide whether to use full combinatorics (age, gender, ethnicity, emotion) or a reduced grid.
   - The current prompt list uses age increments of 5 (432 prompts).
5. **Scale up ratings**:
   - 400 prompts are generated, but only 108 ratings exist. The preference set is still small.
6. **Add evaluation prompts**:
   - Create a held-out set of prompts to compare base LoRA vs RLHF LoRA.

If you want, I can also add a companion `DPO_train.ipynb` or a Python script that reads the manifest and ratings and performs the latent-space DPO training.
