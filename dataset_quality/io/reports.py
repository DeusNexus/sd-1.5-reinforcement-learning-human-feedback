"""Report writers for dataset quality outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _explode_reasons(reasons: pd.Series) -> pd.Series:
    if reasons.empty:
        return reasons
    if reasons.dtype == object and reasons.map(lambda r: isinstance(r, list)).any():
        return reasons.explode()
    return reasons.astype(str).str.split(";").explode()


def write_reports(
    quality_df: pd.DataFrame,
    report_dir: Path,
    worst_metrics: Iterable[str] | None = None,
    worst_n: int = 50,
) -> dict[str, pd.DataFrame]:
    report_dir.mkdir(parents=True, exist_ok=True)

    quality_df.to_csv(report_dir / "quality_report.csv", index=False)
    kept_df = quality_df[quality_df["keep"]].copy()
    dropped_df = quality_df[~quality_df["keep"]].copy()
    kept_df.to_csv(report_dir / "kept_images.csv", index=False)
    dropped_df.to_csv(report_dir / "dropped_images.csv", index=False)

    reasons = _explode_reasons(dropped_df["drop_reasons"])
    reason_counts = reasons.value_counts()
    reason_counts.to_csv(report_dir / "drop_reason_counts.csv", header=["count"])

    if worst_metrics:
        for metric in worst_metrics:
            if metric not in quality_df.columns:
                continue
            df = quality_df.dropna(subset=[metric]).copy()
            df = df.sort_values(metric, ascending=False)
            df.head(worst_n).to_csv(report_dir / f"worst_{metric}.csv", index=False)

    return {"kept": kept_df, "dropped": dropped_df}
