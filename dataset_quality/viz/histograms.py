"""Histogram visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def save_histograms(
    df: pd.DataFrame,
    metrics: Iterable[str],
    output_dir: Path,
    bins: int = 50,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        if metric not in df.columns:
            continue
        series = df[metric].dropna()
        if series.empty:
            continue
        plt.figure()
        series.hist(bins=bins)
        plt.title(metric)
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{metric}.png")
        plt.close()
