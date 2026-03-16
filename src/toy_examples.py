from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .clustering import summarize_clusters, top_down_divisive_clustering
from .kadane_analysis import kadane
from .segmentation import divide_and_conquer_segmentation



def run_toy_examples() -> Dict[str, object]:
    # Segmentation toy example: first half flat, second half volatile
    segmentation_signal = [1, 1, 1, 1, 10, -10, 9, -9]
    segments = divide_and_conquer_segmentation(segmentation_signal, variance_threshold=5.0, min_segment_length=2)

    # Clustering toy example: two obvious groups repeated to 4 clusters through recursive splitting
    features = np.array([
        [0.0, 0.0], [0.1, 0.2], [0.2, 0.1],
        [10.0, 10.0], [10.1, 9.9], [9.8, 10.2],
        [20.0, 20.0], [20.2, 19.9],
        [30.0, 30.0], [29.8, 30.2],
    ])
    classes = ["A", "A", "A", "B", "B", "B", "C", "C", "D", "D"]
    labels = top_down_divisive_clustering(features, n_clusters=4)
    cluster_summary = summarize_clusters(labels, classes)

    # Kadane toy example: strongest sustained gain in middle chunk
    arr = [-2, 1, 2, 3, -1, 2, -5]
    best_sum, start, end = kadane(arr)

    return {
        "segmentation": {
            "signal": segmentation_signal,
            "segment_count": len(segments),
            "segments": [s.to_dict() for s in segments],
        },
        "clustering": {
            "labels": labels.tolist(),
            "cluster_summary": cluster_summary.to_dict(orient="records"),
        },
        "kadane": {
            "array": arr,
            "best_sum": float(best_sum),
            "start": int(start),
            "end": int(end),
            "expected_subarray": arr[start : end + 1],
        },
    }
