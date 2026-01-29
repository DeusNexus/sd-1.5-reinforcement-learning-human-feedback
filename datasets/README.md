PUT dataset folder here -> datasets/appa-real-dataset_v2

```bash
.
├── datasets
│   └── appa-real-dataset_v2
│       ├── test_data
│       ├── train_data
│       └── valid_data
│       └── labels_metadata_test.csv
│       └── labels_metadata_train.csv
│       └── labels_metadata_valid.csv
```

## Dataset Cleaning (appa-real-dataset_v2_improved)

We generate a cleaned variant in `datasets/appa-real-dataset_v2_improved` via the
`dataset_quality` pipeline. The goal is to remove clearly low‑quality images
while keeping class balance across ages, genders, ethnicities, and emotions.

### What gets measured per image
- **Sharpness (ROI + global):** Laplacian variance and Tenengrad.
- **Compression artifacts:** JPEG blockiness and an artifact score.
- **NR‑IQA:** NIQE and BRISQUE.
- **Black bars:** score + area/shape heuristics.
- **Face detection:** used to define ROI; fallback ROI if no face detected.

### How images are dropped
1) **Hard thresholds (quality floor):**  
   Immediate drops for blur, high blockiness, high NIQE/BRISQUE, or strong black bars.
   These are the “non‑negotiable” bad images.

2) **Worst‑percentile pruning (soft cleanup):**  
   We compute a weighted badness score across multiple metrics and remove the
   worst X% **within each 5‑year age bin** to avoid skewing rare ages.
   - Config: `drop_worst_frac = 0.2`
   - Stratified by age bin size = 5 years
   - **Minimum keep per age bin = 200** (only for the percentile step)
   - Hard‑threshold failures are still dropped even if a bin falls below 200.

### Where this is configured
- `dataset_quality/config.py`  
  - `THRESH`: hard cutoffs  
  - `PIPELINE.drop_worst_*`: percentile pruning and age‑bin stratification

### Outputs
- Cleaned dataset: `datasets/appa-real-dataset_v2_improved/`
- Reports: `datasets/appa-real-dataset_v2_improved/reports/`
  - `kept_images.csv`, `dropped_images.csv`, plus metric summaries
