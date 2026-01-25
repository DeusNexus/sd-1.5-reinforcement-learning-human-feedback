# Implementation Plan

This document lists what still needs to be done to complete the RLHF + latent-space DPO pipeline, based on the code and artifacts currently in the repo.

## Goal
Train a Stable Diffusion LoRA that better matches prompts describing age, gender, ethnicity, and emotion, using human preference data and a latent-space DPO loss.

## Completed Components (already in repo)
- Base LoRA training pipeline in `1_diffusion_rl_base.ipynb` and `1_diffusion_rl_base_colab_v2.ipynb`.
- Prompt generation utility in `2_generate_prompts.ipynb`.
- RLHF image + latent generation in `3_1_diffusion_sample_rlhf.ipynb` and `3_1_diffusion_sample_rlhf_colab.ipynb`.
- Human preference UI in `4_hf_rating.ipynb`.
- Preference data and manifests in `ratings_interface/`.

## Remaining Implementation Tasks

### 1) Finalize prompt grid
- Decide the authoritative prompt grid (full vs reduced).
- If using the full grid, regenerate `prompts_*.txt` using `2_generate_prompts.ipynb`.
- Ensure prompt file used by `3_1_diffusion_sample_rlhf.ipynb` is the intended one.

### 2) Decide latent representation to store
Current generation saves **initial noise latents**. The DPO plan sometimes describes saving the **final denoised latent (x0)**.
- Pick one representation and standardize it.
- Update `3_1_diffusion_sample_rlhf.ipynb` to store the chosen latent.
- Update the manifest to record the latent type.

### 3) Expand manifest metadata for reproducibility
Add to `ratings_interface/rlhf_generation_data/generation_manifest.jsonl`:
- `lora_run_id` and `weight_name`
- scheduler name or config
- base model version
- latent type (noise or x0)

### 4) Implement DPO dataset loader
Create a loader that joins:
- `ratings_interface/ratings_real.jsonl`
- `ratings_interface/rlhf_generation_data/generation_manifest.jsonl`

The loader should return:
- tokenized prompt
- preferred latent path
- rejected latent path
- generation metadata (steps, guidance, seed)

### 5) Implement latent-space DPO loss
Add a training loop that:
- Loads the base SD pipeline + LoRA.
- Freezes VAE + text encoder.
- Runs UNet forward on preferred and rejected latents.
- Computes a preference loss (DPO-style).
- Updates only LoRA weights.

Optional refinement:
- Contrastive loss on UNet hidden states for semantic alignment.

### 6) Training output and tracking
- Save best LoRA adapter and full checkpoint.
- Save training history (loss, beta, steps).
- Write a small evaluation script to compare base vs RLHF LoRA.

### 7) Evaluation and iteration
- Create a held-out prompt set for evaluation.
- Compare preference win rate or manual ratings.
- Iterate: regenerate samples -> re-rate -> re-train.

### Dataset quality (new pipeline in place)
- Use `dataset_quality/` pipeline and review notebooks to curate a cleaned dataset before retraining LoRA.

## Suggested Deliverables
- `dpo_train.ipynb` or `dpo_train.py`
- `dpo_dataset.py` (loader)
- `eval_prompts.txt` (held-out prompts)
- Updated `REPORT.md` once DPO is in place
