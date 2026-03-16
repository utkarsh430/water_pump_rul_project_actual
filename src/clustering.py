from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import numpy as np
import pandas as pd


def farthest_pair(points: np.ndarray) -> tuple[int, int]:
    """Approximate farthest pair using a two-sweep heuristic.

    This avoids the O(n^2) memory/time cost of a full pairwise distance matrix,
    which is impractical for 10,000 rows in this project environment.
    """
    n = len(points)
    if n < 2:
        return 0, 0

    start = 0
    d0 = np.sum((points - points[start]) ** 2, axis=1)
    a = int(np.argmax(d0))
    da = np.sum((points - points[a]) ** 2, axis=1)
    b = int(np.argmax(da))
    return a, b



def top_down_divisive_clustering(features: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    n_samples = features.shape[0]
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_samples == 0:
        raise ValueError("No samples provided")
    if n_clusters >= n_samples:
        return np.arange(n_samples)

    clusters: List[np.ndarray] = [np.arange(n_samples)]

    while len(clusters) < n_clusters:
        split_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        selected = clusters.pop(split_idx)
        if len(selected) <= 1:
            clusters.append(selected)
            break

        subset = features[selected]
        a_local, b_local = farthest_pair(subset)
        pivot_a = subset[a_local]
        pivot_b = subset[b_local]

        left: List[int] = []
        right: List[int] = []
        for global_idx in selected:
            point = features[global_idx]
            da = float(np.linalg.norm(point - pivot_a))
            db = float(np.linalg.norm(point - pivot_b))
            if da <= db:
                left.append(int(global_idx))
            else:
                right.append(int(global_idx))

        if not left or not right:
            midpoint = len(selected) // 2
            left = selected[:midpoint].tolist()
            right = selected[midpoint:].tolist()

        clusters.append(np.array(left, dtype=int))
        clusters.append(np.array(right, dtype=int))

    labels = np.empty(n_samples, dtype=int)
    for cluster_id, members in enumerate(clusters[:n_clusters]):
        labels[members] = cluster_id
    return labels



def summarize_clusters(labels: np.ndarray, true_classes: Sequence[str]) -> pd.DataFrame:
    rows = []
    classes = np.asarray(true_classes)
    for cluster_id in sorted(np.unique(labels)):
        members = classes[labels == cluster_id]
        counts = Counter(members)
        majority_class, majority_count = counts.most_common(1)[0]
        row = {
            "cluster_id": int(cluster_id),
            "size": int(len(members)),
            "majority_class": majority_class,
            "majority_count": int(majority_count),
            "purity": float(majority_count / len(members)),
        }
        for klass, count in counts.items():
            row[f"count::{klass}"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
