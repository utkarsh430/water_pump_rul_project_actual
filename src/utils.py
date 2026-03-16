from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


RUL_LABELS = [
    "Extremely Low RUL",
    "Moderately Low RUL",
    "Moderately High RUL",
    "Extremely High RUL",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_rul_thresholds(rul: pd.Series, q_low: float = 0.10, q_mid: float = 0.50, q_high: float = 0.90) -> Dict[str, float]:
    clean = rul.dropna().astype(float)
    if clean.empty:
        raise ValueError("RUL column is empty after dropping missing values.")
    return {
        "Q10": float(clean.quantile(q_low)),
        "Q50": float(clean.quantile(q_mid)),
        "Q90": float(clean.quantile(q_high)),
    }



def assign_rul_category(value: float, thresholds: Dict[str, float]) -> str:
    if value < thresholds["Q10"]:
        return "Extremely Low RUL"
    if value < thresholds["Q50"]:
        return "Moderately Low RUL"
    if value < thresholds["Q90"]:
        return "Moderately High RUL"
    return "Extremely High RUL"



def add_rul_categories(df: pd.DataFrame, rul_col: str = "rul") -> Tuple[pd.DataFrame, Dict[str, float]]:
    thresholds = compute_rul_thresholds(df[rul_col])
    enriched = df.copy()
    enriched["rul_category"] = enriched[rul_col].apply(lambda x: assign_rul_category(float(x), thresholds))
    return enriched, thresholds



def systematic_sensor_selection(sensor_columns: Sequence[str], k: int = 10) -> List[str]:
    if k <= 0:
        raise ValueError("k must be positive.")
    sensors = list(sensor_columns)
    if len(sensors) <= k:
        return sensors
    indices = np.linspace(0, len(sensors) - 1, k, dtype=int)
    return [sensors[i] for i in indices]



def to_pretty_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")



def safe_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return float(value)
