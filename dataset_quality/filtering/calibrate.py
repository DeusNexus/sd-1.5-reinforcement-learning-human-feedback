"""Calibration utilities for threshold tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


LOW_IS_WORSE = {"roi_tenengrad", "roi_lap_var", "tenengrad", "lap_var"}


def metric_percentiles(
    df: pd.DataFrame,
    metric_cols: Iterable[str],
    percentiles: Iterable[float] = (0.05, 0.1, 0.9, 0.95),
) -> pd.DataFrame:
    return df[list(metric_cols)].quantile(list(percentiles))


def build_calibration_pack(
    df: pd.DataFrame,
    output_csv: Path,
    metrics: Iterable[str],
    worst_n: int = 50,
    random_n: int = 50,
) -> pd.DataFrame:
    selections = []

    for metric in metrics:
        if metric not in df.columns:
            continue
        subset = df.dropna(subset=[metric]).copy()
        ascending = metric in LOW_IS_WORSE
        subset = subset.sort_values(metric, ascending=ascending).head(worst_n)
        subset = subset.assign(selection_reason=f"worst_{metric}")
        selections.append(subset)

    if "keep" in df.columns:
        kept = df[df["keep"]].sample(n=min(random_n, len(df[df["keep"]])), random_state=0)
        kept = kept.assign(selection_reason="random_kept")
        selections.append(kept)
        dropped = df[~df["keep"]].sample(
            n=min(random_n, len(df[~df["keep"]])),
            random_state=0,
        )
        dropped = dropped.assign(selection_reason="random_dropped")
        selections.append(dropped)

    if selections:
        pack = pd.concat(selections, ignore_index=True)
    else:
        pack = df.copy()

    pack = pack.drop_duplicates(subset=["image_path"], keep="first")
    pack["label"] = ""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pack.to_csv(output_csv, index=False)
    return pack
