from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class KadaneResult:
    sensor: str
    start_index: int
    end_index: int
    total_deviation: float
    dominant_rul_category: str
    dominant_count: int
    interval_length: int

    def to_dict(self) -> dict:
        return {
            "sensor": self.sensor,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "total_deviation": float(self.total_deviation),
            "dominant_rul_category": self.dominant_rul_category,
            "dominant_count": int(self.dominant_count),
            "interval_length": int(self.interval_length),
        }



def kadane(arr: Sequence[float]) -> tuple[float, int, int]:
    best_sum = float("-inf")
    current_sum = 0.0
    best_start = best_end = current_start = 0

    for i, value in enumerate(arr):
        if current_sum <= 0:
            current_sum = float(value)
            current_start = i
        else:
            current_sum += float(value)

        if current_sum > best_sum:
            best_sum = current_sum
            best_start = current_start
            best_end = i

    return best_sum, best_start, best_end



def analyze_sensor_with_kadane(df: pd.DataFrame, sensor_col: str, rul_category_col: str = "rul_category") -> KadaneResult:
    signal = df[sensor_col].to_numpy(dtype=float)
    if len(signal) < 2:
        raise ValueError(f"Sensor {sensor_col} must have at least two values.")

    absolute_diff = np.abs(np.diff(signal))
    centered = absolute_diff - absolute_diff.mean()
    best_sum, start, end = kadane(centered)

    interval_categories = df.iloc[start : end + 2][rul_category_col].tolist()
    counts = Counter(interval_categories)
    dominant_rul_category, dominant_count = counts.most_common(1)[0]

    return KadaneResult(
        sensor=sensor_col,
        start_index=int(start),
        end_index=int(end + 1),
        total_deviation=float(best_sum),
        dominant_rul_category=dominant_rul_category,
        dominant_count=int(dominant_count),
        interval_length=int(end - start + 2),
    )
