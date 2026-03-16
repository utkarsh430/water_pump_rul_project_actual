from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .utils import add_rul_categories


REQUIRED_BASE_COLUMNS = {"timestamp", "rul"}


class DatasetValidationError(Exception):
    pass



def load_dataset(csv_path: str | Path, max_rows: int = 10_000) -> Tuple[pd.DataFrame, List[str], dict]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise DatasetValidationError("Dataset is empty.")

    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    missing_base = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
    if missing_base:
        raise DatasetValidationError(
            f"Dataset must contain columns {sorted(REQUIRED_BASE_COLUMNS)}. Missing: {missing_base}"
        )

    sensor_columns = [c for c in df.columns if c.startswith("sensor_")]
    if len(sensor_columns) < 10:
        raise DatasetValidationError(
            f"Expected many sensor columns like sensor_00 ... sensor_51, found only {len(sensor_columns)}."
        )

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().all():
        df["timestamp"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    numeric_cols = ["rul"] + sensor_columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["rul"]).reset_index(drop=True)
    df[sensor_columns] = df[sensor_columns].interpolate(limit_direction="both")
    df[sensor_columns] = df[sensor_columns].fillna(df[sensor_columns].median(numeric_only=True))

    df, thresholds = add_rul_categories(df, rul_col="rul")
    return df, sensor_columns, thresholds
