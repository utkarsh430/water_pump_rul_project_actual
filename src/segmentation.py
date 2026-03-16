from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class Segment:
    start: int
    end: int
    variance: float
    length: int
    is_stable: bool

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "variance": float(self.variance),
            "length": self.length,
            "is_stable": self.is_stable,
        }



def segment_variance(signal: Sequence[float], start: int, end: int) -> float:
    piece = np.asarray(signal[start:end], dtype=float)
    if len(piece) <= 1:
        return 0.0
    return float(np.var(piece))



def divide_and_conquer_segmentation(
    signal: Sequence[float],
    variance_threshold: float,
    min_segment_length: int = 32,
    start: int = 0,
    end: int | None = None,
) -> List[Segment]:
    if end is None:
        end = len(signal)
    if end - start <= 0:
        return []

    current_var = segment_variance(signal, start, end)
    length = end - start

    if length <= min_segment_length or current_var <= variance_threshold:
        return [Segment(start, end, current_var, length, True)]

    mid = (start + end) // 2
    if mid == start or mid == end:
        return [Segment(start, end, current_var, length, True)]

    left = divide_and_conquer_segmentation(signal, variance_threshold, min_segment_length, start, mid)
    right = divide_and_conquer_segmentation(signal, variance_threshold, min_segment_length, mid, end)
    return left + right



def segmentation_complexity(segments: List[Segment]) -> int:
    return len(segments)
