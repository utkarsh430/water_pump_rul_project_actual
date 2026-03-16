from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.clustering import summarize_clusters, top_down_divisive_clustering
from src.data_loader import load_dataset
from src.kadane_analysis import analyze_sensor_with_kadane
from src.plotting import plot_kadane_interval, plot_segmentation
from src.segmentation import divide_and_conquer_segmentation, segmentation_complexity
from src.toy_examples import run_toy_examples
from src.utils import ensure_dir, systematic_sensor_selection, to_pretty_json


def choose_variance_threshold(signal: pd.Series, multiplier: float = 0.50) -> float:
    signal_var = float(np.var(signal.to_numpy(dtype=float)))
    return max(signal_var * multiplier, 1e-9)


def task1_segmentation(df: pd.DataFrame, selected_sensors: List[str], output_dir: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    seg_plot_dir = output_dir / "task1_segmentation_plots"
    ensure_dir(seg_plot_dir)

    for sensor in selected_sensors:
        threshold = choose_variance_threshold(df[sensor])
        segments = divide_and_conquer_segmentation(
            df[sensor].to_numpy(dtype=float),
            variance_threshold=threshold,
            min_segment_length=max(16, len(df) // 64),
        )
        complexity = segmentation_complexity(segments)
        plot_segmentation(df, sensor, segments, seg_plot_dir / f"{sensor}_segmentation.png")

        segment_rul_labels = []
        for seg in segments:
            rul_slice = df.iloc[seg.start:seg.end]["rul_category"]
            dominant_rul = rul_slice.mode().iloc[0]
            segment_rul_labels.append(dominant_rul)

        rul_distribution = pd.Series(segment_rul_labels).value_counts().to_dict()

        if complexity >= 12:
            temporal_comment = "Highly dynamic signal with frequent structural changes."
        elif complexity >= 6:
            temporal_comment = "Moderately dynamic signal with noticeable regime changes."
        else:
            temporal_comment = "Relatively stable signal with fewer regime shifts."

        low_rul_segments = int(rul_distribution.get("Extremely Low RUL", 0) + rul_distribution.get("Moderately Low RUL", 0))
        if low_rul_segments >= max(1, complexity // 2):
            rul_comment = "A substantial share of segments falls in lower-RUL regions, suggesting degradation sensitivity."
        elif low_rul_segments > 0:
            rul_comment = "Some segments overlap lower-RUL regions, but the relationship is mixed."
        else:
            rul_comment = "Segments are concentrated outside lower-RUL regions."

        results.append(
            {
                "sensor": sensor,
                "variance_threshold": float(threshold),
                "segment_count": int(complexity),
                "avg_segment_length": float(np.mean([s.length for s in segments])),
                "max_segment_variance": float(max(s.variance for s in segments)),
                "segment_rul_distribution": rul_distribution,
                "temporal_dynamics_discussion": temporal_comment,
                "rul_relation_discussion": rul_comment,
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_dir / "task1_segmentation_summary.csv", index=False)
    result_df.sort_values("segment_count", ascending=False)[["sensor", "segment_count"]].to_csv(
        output_dir / "task1_segmentation_ranking.csv", index=False
    )
    return results


def task2_clustering(df: pd.DataFrame, sensor_columns: List[str], output_dir: Path) -> List[Dict[str, object]]:
    features = df[sensor_columns].to_numpy(dtype=float)
    features = (features - features.mean(axis=0)) / np.where(features.std(axis=0) == 0, 1.0, features.std(axis=0))
    labels = top_down_divisive_clustering(features, n_clusters=4)
    cluster_summary = summarize_clusters(labels, df["rul_category"].tolist())
    cluster_summary["mapping_discussion"] = cluster_summary.apply(
        lambda row: f"Cluster {int(row['cluster_id'])} is dominated by {row['majority_class']} with purity {row['purity']:.3f}.",
        axis=1,
    )
    cluster_summary.to_csv(output_dir / "task2_clustering_summary.csv", index=False)
    labelled = df[["timestamp", "rul", "rul_category"] + sensor_columns].copy()
    labelled["cluster_id"] = labels
    labelled.to_csv(output_dir / "task2_cluster_assignments.csv", index=False)
    return cluster_summary.to_dict(orient="records")


def task3_kadane(df: pd.DataFrame, sensor_columns: List[str], output_dir: Path) -> List[Dict[str, object]]:
    results = []
    kadane_plot_dir = output_dir / "task3_kadane_plots"
    ensure_dir(kadane_plot_dir)

    for sensor in sensor_columns:
        result = analyze_sensor_with_kadane(df, sensor)
        results.append(result.to_dict())
        plot_kadane_interval(df, sensor, result.start_index, result.end_index, kadane_plot_dir / f"{sensor}_kadane.png")

    result_df = pd.DataFrame(results)
    low_rul_labels = {"Extremely Low RUL", "Moderately Low RUL"}
    result_df["is_low_rul_indicator"] = result_df["dominant_rul_category"].isin(low_rul_labels)
    result_df = result_df.sort_values(["is_low_rul_indicator", "total_deviation"], ascending=[False, False])
    result_df.to_csv(output_dir / "task3_kadane_summary.csv", index=False)
    result_df[result_df["is_low_rul_indicator"]].to_csv(output_dir / "task3_early_warning_sensors.csv", index=False)
    return result_df.to_dict(orient="records")


def build_summary(df: pd.DataFrame, sensor_columns: List[str], thresholds: Dict[str, float], selected_sensors: List[str], task1, task2, task3) -> Dict[str, object]:
    missing_prompt_sensors = [s for s in [f"sensor_{i:02d}" for i in range(52)] if s not in sensor_columns]
    return {
        "dataset": {
            "rows_used": int(len(df)),
            "sensor_count": int(len(sensor_columns)),
            "selected_sensors": selected_sensors,
            "thresholds": thresholds,
            "rul_category_counts": df["rul_category"].value_counts().to_dict(),
            "available_sensor_columns": sensor_columns,
            "missing_prompt_sensors": missing_prompt_sensors,
        },
        "toy_examples": run_toy_examples(),
        "task1_segmentation": task1,
        "task2_clustering": task2,
        "task3_kadane": task3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Water Pump RUL algorithmic analysis project")
    parser.add_argument("--csv_path", required=True, help="Path to the Kaggle CSV dataset")
    parser.add_argument("--output_dir", default="output", help="Folder where outputs will be saved")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df, sensor_columns, thresholds = load_dataset(args.csv_path, max_rows=10_000)
    selected_sensors = systematic_sensor_selection(sensor_columns, k=10)

    task1 = task1_segmentation(df, selected_sensors, output_dir)
    task2 = task2_clustering(df, sensor_columns, output_dir)
    task3 = task3_kadane(df, sensor_columns, output_dir)

    summary = build_summary(df, sensor_columns, thresholds, selected_sensors, task1, task2, task3)
    to_pretty_json(summary, output_dir / "summary.json")
    print(f"Analysis complete. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
