from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from .segmentation import Segment



def plot_segmentation(df: pd.DataFrame, sensor_col: str, segments: List[Segment], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(len(df))
    y = df[sensor_col].to_numpy()
    ax.plot(x, y, linewidth=1.0)
    for seg in segments:
        ax.axvspan(seg.start, seg.end - 1, alpha=0.15)
        ax.axvline(seg.start, linestyle='--', linewidth=0.8)
    ax.set_title(f"Divide-and-Conquer Segmentation - {sensor_col}")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Sensor Value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



def plot_kadane_interval(df: pd.DataFrame, sensor_col: str, start_index: int, end_index: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(len(df))
    y = df[sensor_col].to_numpy()
    ax.plot(x, y, linewidth=1.0)
    ax.axvspan(start_index, end_index, alpha=0.2)
    ax.set_title(f"Maximum-Deviation Interval (Kadane) - {sensor_col}")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Sensor Value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
