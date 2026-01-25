"""Dataset output helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from dataset_quality import config


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        if not dst.exists():
            os.link(src, dst)
    elif mode == "symlink":
        if not dst.exists():
            os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def write_split_csvs(
    df: pd.DataFrame,
    output_root: Path,
    split_dirs: dict[str, str] | None = None,
) -> None:
    split_dirs = split_dirs or config.SPLIT_DIRS
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dir in split_dirs.items():
        split_df = df[df["split"] == split_name]
        csv_out = output_root / f"labels_metadata_{split_name}.csv"
        split_df[
            ["imageId", "age", "gender", "ethnicity", "emotion"]
        ].to_csv(csv_out, index=False)
        (output_root / split_dir).mkdir(parents=True, exist_ok=True)


def write_images(
    df: pd.DataFrame,
    output_root: Path,
    link_mode: str,
    split_dirs: dict[str, str] | None = None,
    image_path_col: str = "image_path",
) -> None:
    split_dirs = split_dirs or config.SPLIT_DIRS
    for split_name, split_dir in split_dirs.items():
        split_df = df[df["split"] == split_name]
        target_dir = output_root / split_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for row in split_df.itertuples(index=False):
            src = Path(getattr(row, image_path_col))
            dst = target_dir / f"{int(row.imageId):06d}.jpg"
            link_or_copy(src, dst, link_mode)


def write_dataset(
    df: pd.DataFrame,
    output_root: Path,
    link_mode: str | None = None,
    split_dirs: dict[str, str] | None = None,
    image_path_col: str = "image_path",
) -> None:
    link_mode = link_mode or config.PIPELINE["link_mode"]
    write_split_csvs(df, output_root, split_dirs=split_dirs)
    write_images(
        df,
        output_root,
        link_mode=link_mode,
        split_dirs=split_dirs,
        image_path_col=image_path_col,
    )
