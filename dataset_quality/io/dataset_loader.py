"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from dataset_quality import config


def load_splits(
    dataset_root: Path,
    splits: Iterable[str] | None = None,
    split_dirs: dict[str, str] | None = None,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    split_dirs = split_dirs or config.SPLIT_DIRS
    splits = list(splits) if splits is not None else ["train", "valid", "test"]

    frames = []
    for split_name in splits:
        csv_path = dataset_root / f"labels_metadata_{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df["split"] = split_name
        df["image_path"] = df["imageId"].apply(
            lambda x: dataset_root / split_dirs[split_name] / f"{int(x):06d}.jpg"
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
