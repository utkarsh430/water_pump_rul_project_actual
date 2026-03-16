# Algorithmic Time-Series Segmentation and Condition Analysis Using Water Pump RUL Dataset

## 1. Project Overview
This project analyzes the first 10,000 rows of the Water Pump RUL dataset (https://www.kaggle.com/datasets/anseldsouza/water-pump-rul-predictive-maintenance) using from-scratch algorithms only.
The goal is to analyze temporal sensor behavior, group similar operating states, and identify sensors that may act as early indicators of low remaining useful life.
 The implementation covers:
- Divide-and-Conquer segmentation on 10 selected sensors.
- Recursive top-down clustering into 4 clusters on all available sensor columns in the dataset.
- Maximum subarray analysis (Kadane) on all available sensors after absolute first-difference preprocessing.

No machine-learning libraries are used in the core algorithms.

## 2. Installation and Usage Instructions
1. Create a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run:
   ```bash
   python main.py --csv_path "PATH_TO_YOUR_DATASET.csv"
   ```
4. All generated outputs are saved inside the chosen output folder.

## 3. Code Structure Explanation
- `main.py`: orchestration entry point.
- `src/data_loader.py`: dataset loading, cleaning, and RUL categorization.
- `src/segmentation.py`: divide-and-conquer segmentation.
- `src/clustering.py`: recursive top-down clustering from scratch.
- `src/kadane_analysis.py`: maximum subarray analysis.
- `src/toy_examples.py`: tiny correctness checks.
- `src/plotting.py`: output figures.


## 4. Algorithm Descriptions
### 4.1 Divide-and-Conquer Segmentation
For a current segment, compute variance. If the variance is above a threshold and the segment is larger than the minimum segment length, split it into left and right halves recursively. Otherwise, mark the segment as stable.

### 4.2 Divide-and-Conquer Clustering
Starting from all 10,000 time instances in one cluster, repeatedly split the largest current cluster until 4 clusters are obtained. A split is made using a from-scratch farthest-pivot rule. Each point is assigned to the nearer pivot.

### 4.3 Maximum Subarray (Kadane)
For each sensor, compute absolute first-differences and center them by subtracting their mean. Kadane's algorithm then finds the maximum-sum contiguous interval, which identifies the strongest sustained deviation period.

## 5. Toy Example Verification
### Segmentation
Segment count: 3
Segments: [{'start': 0, 'end': 4, 'variance': 0.0, 'length': 4, 'is_stable': True}, {'start': 4, 'end': 6, 'variance': 100.0, 'length': 2, 'is_stable': True}, {'start': 6, 'end': 8, 'variance': 81.0, 'length': 2, 'is_stable': True}]

### Clustering
|   cluster_id |   size | majority_class   |   majority_count |   purity |   count::B |   count::A |   count::D |   count::C |
|-------------:|-------:|:-----------------|-----------------:|---------:|-----------:|-----------:|-----------:|-----------:|
|            0 |      3 | B                |                3 |        1 |          3 |        0 |        0 |        0 |
|            1 |      3 | A                |                3 |        1 |        0 |          3 |        0 |        0 |
|            2 |      2 | D                |                2 |        1 |        0 |        0 |          2 |        0 |
|            3 |      2 | C                |                2 |        1 |        0 |        0 |        0 |          2 |

### Kadane
Best sum: 7.0
Subarray interval: [1, 5]
Subarray: [1, 2, 3, -1, 2]

## 6. Dataset Description
- Rows used: 10000
- Available sensor columns used for clustering: 50
- Selected 10 sensors for segmentation: sensor_00, sensor_05, sensor_10, sensor_17, sensor_22, sensor_28, sensor_33, sensor_39, sensor_44, sensor_51
- RUL thresholds: Q10 = 135.9317, Q50 = 202.5917, Q90 = 269.2517
- Variables used in this project were `timestamp`, all available `sensor_*` columns, the original `rul` column, and the derived `rul_category` variable created using the Q10, Q50, and Q90 thresholds.
- RUL category counts:

| rul_category        |   count |
|:--------------------|--------:|
| Moderately High RUL |    4000 |
| Moderately Low RUL  |    4000 |
| Extremely High RUL  |    1000 |
| Extremely Low RUL   |    1000 |

## 7. Execution Results
### Task 1 - Segmentation
| sensor    |   variance_threshold |   segment_count |   avg_segment_length |   max_segment_variance | segment_rul_distribution                                                                               | temporal_dynamics_discussion                              | rul_relation_discussion                                                                         |
|:----------|---------------------:|----------------:|---------------------:|-----------------------:|:-------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|:------------------------------------------------------------------------------------------------|
| sensor_00 |           0.00064932 |              12 |              833.333 |             0.00414727 | {'Moderately High RUL': 5, 'Extremely Low RUL': 5, 'Moderately Low RUL': 2}                            | Highly dynamic signal with frequent structural changes.   | A substantial share of segments falls in lower-RUL regions, suggesting degradation sensitivity. |
| sensor_05 |          11.9491     |              44 |              227.273 |            45.321      | {'Moderately Low RUL': 20, 'Moderately High RUL': 17, 'Extremely Low RUL': 6, 'Extremely High RUL': 1} | Highly dynamic signal with frequent structural changes.   | A substantial share of segments falls in lower-RUL regions, suggesting degradation sensitivity. |
| sensor_10 |           4.55621    |              60 |              166.667 |            23.244      | {'Moderately Low RUL': 27, 'Moderately High RUL': 24, 'Extremely Low RUL': 8, 'Extremely High RUL': 1} | Highly dynamic signal with frequent structural changes.   | A substantial share of segments falls in lower-RUL regions, suggesting degradation sensitivity. |
| sensor_17 |          22.8397     |              60 |              166.667 |           596.201      | {'Moderately High RUL': 26, 'Moderately Low RUL': 25, 'Extremely Low RUL': 8, 'Extremely High RUL': 1} | Highly dynamic signal with frequent structural changes.   | A substantial share of segments falls in lower-RUL regions, suggesting degradation sensitivity. |
| sensor_22 |         142.37       |               9 |             1111.11  |           450.56       | {'Extremely High RUL': 4, 'Moderately High RUL': 4, 'Moderately Low RUL': 1}                           | Moderately dynamic signal with noticeable regime changes. | Some segments overlap lower-RUL regions, but the relationship is mixed.                         |
| sensor_28 |        1950.13       |               7 |             1428.57  |        101926          | {'Moderately Low RUL': 6, 'Moderately High RUL': 1}                                                    | Moderately dynamic signal with noticeable regime changes. | A substantial share of segments falls in lower-RUL regions, suggesting degradation sensitivity. |
| sensor_33 |        2533.82       |               6 |             1666.67  |          2392.22       | {'Moderately Low RUL': 4, 'Moderately High RUL': 2}                                                    | Moderately dynamic signal with noticeable regime changes. | A substantial share of segments falls in lower-RUL regions, suggesting degradation sensitivity. |
| sensor_39 |          12.0118     |              22 |              454.545 |           275.766      | {'Moderately Low RUL': 12, 'Moderately High RUL': 8, 'Extremely High RUL': 1, 'Extremely Low RUL': 1}  | Highly dynamic signal with frequent structural changes.   | A substantial share of segments falls in lower-RUL regions, suggesting degradation sensitivity. |
| sensor_44 |          28.2919     |              46 |              217.391 |           126.176      | {'Moderately High RUL': 21, 'Moderately Low RUL': 19, 'Extremely High RUL': 5, 'Extremely Low RUL': 1} | Highly dynamic signal with frequent structural changes.   | Some segments overlap lower-RUL regions, but the relationship is mixed.                         |
| sensor_51 |         354.201      |               9 |             1111.11  |          1692.93       | {'Moderately Low RUL': 8, 'Moderately High RUL': 1}                                                    | Moderately dynamic signal with noticeable regime changes. | A substantial share of segments falls in lower-RUL regions, suggesting degradation sensitivity. |

Interpretation:
The segmentation complexity score equals the number of final recursive segments. Higher scores indicate more frequent structural changes in the signal. In the actual execution, `sensor_10` and `sensor_17` showed the highest segmentation complexity with 60 segments each, while `sensor_33` and `sensor_28` were among the most stable with 6 and 7 segments respectively. The segment-level dominant RUL categories also suggest that several highly segmented sensors overlap substantially with lower-RUL regions, indicating a possible relationship between temporal instability and degradation.

### Task 2 - Clustering
|   cluster_id |   size | majority_class      |   majority_count |   purity |   count::Extremely High RUL |   count::Moderately High RUL |   count::Moderately Low RUL |   count::Extremely Low RUL | mapping_discussion                                               |
|-------------:|-------:|:--------------------|-----------------:|---------:|----------------------------:|-----------------------------:|----------------------------:|---------------------------:|:-----------------------------------------------------------------|
|            0 |     77 | Moderately High RUL |               39 | 0.506494 |                           1 |                           39 |                          31 |                          6 | Cluster 0 is dominated by Moderately High RUL with purity 0.506. |
|            1 |     11 | Moderately Low RUL  |               10 | 0.909091 |                         nan |                            1 |                          10 |                        nan | Cluster 1 is dominated by Moderately Low RUL with purity 0.909.  |
|            2 |     52 | Moderately Low RUL  |               47 | 0.903846 |                         nan |                            2 |                          47 |                          3 | Cluster 2 is dominated by Moderately Low RUL with purity 0.904.  |
|            3 |   9860 | Moderately High RUL |             3958 | 0.40142  |                         999 |                         3958 |                        3912 |                        991 | Cluster 3 is dominated by Moderately High RUL with purity 0.401. |

Interpretation:
Each cluster is mapped to the dominant true RUL class shown above, together with purity and class counts. Higher purity means stronger alignment between the unsupervised split and the RUL-based health class.

### Task 3 - Maximum Subarray (Kadane)
| sensor    |   start_index |   end_index |   total_deviation | dominant_rul_category   |   dominant_count |   interval_length | is_low_rul_indicator   |
|:----------|--------------:|------------:|------------------:|:------------------------|-----------------:|------------------:|:-----------------------|
| sensor_32 |          4197 |        8180 |          5165.27  | Moderately Low RUL      |             3181 |              3984 | True                   |
| sensor_29 |          5061 |        9999 |          4935.52  | Moderately Low RUL      |             3939 |              4939 | True                   |
| sensor_17 |          7630 |        9999 |          4372.13  | Moderately Low RUL      |             1370 |              2370 | True                   |
| sensor_37 |          8808 |        9999 |          4352.52  | Extremely Low RUL       |             1000 |              1192 | True                   |
| sensor_25 |          5332 |        9998 |          4152.69  | Moderately Low RUL      |             3668 |              4667 | True                   |
| sensor_36 |          3545 |        9999 |          4095.08  | Moderately Low RUL      |             4000 |              6455 | True                   |
| sensor_24 |          8600 |        9999 |          3682.63  | Extremely Low RUL       |             1000 |              1400 | True                   |
| sensor_35 |          3450 |        9998 |          2681.12  | Moderately Low RUL      |             4000 |              6549 | True                   |
| sensor_40 |          7332 |        9113 |          1776.44  | Moderately Low RUL      |             1668 |              1782 | True                   |
| sensor_48 |          5586 |        7603 |          1321.92  | Moderately Low RUL      |             2018 |              2018 | True                   |
| sensor_22 |          8599 |        9998 |          1209.99  | Extremely Low RUL       |              999 |              1400 | True                   |
| sensor_46 |          7268 |        9783 |          1056.75  | Moderately Low RUL      |             1732 |              2516 | True                   |
| sensor_13 |          5857 |        9975 |          1024.12  | Moderately Low RUL      |             3143 |              4119 | True                   |
| sensor_43 |          7337 |        9300 |           880.277 | Moderately Low RUL      |             1663 |              1964 | True                   |
| sensor_19 |          8717 |        9999 |           687.985 | Extremely Low RUL       |             1000 |              1283 | True                   |
| sensor_51 |          4577 |        7869 |           630.025 | Moderately Low RUL      |             2870 |              3293 | True                   |
| sensor_14 |          8622 |        9995 |           354.074 | Extremely Low RUL       |              996 |              1374 | True                   |
| sensor_47 |          4852 |        9897 |           325.672 | Moderately Low RUL      |             4000 |              5046 | True                   |
| sensor_10 |          2786 |        9995 |           313.19  | Moderately Low RUL      |             4000 |              7210 | True                   |
| sensor_38 |          3110 |        9843 |           276.132 | Moderately Low RUL      |             4000 |              6734 | True                   |

Interpretation:
For each sensor, Kadane's algorithm identifies the strongest sustained deviation interval. In the actual dataset, sensors such as `sensor_32`, `sensor_29`, `sensor_17`, and `sensor_37` emerged as particularly strong early-warning candidates because their maximum-deviation intervals were dominated by Moderately Low RUL or Extremely Low RUL and had relatively large total deviation values.

## 8. Discussion and Conclusions
### Interpret Findings
- The segmentation results show that temporal instability differs substantially across sensors. For example, `sensor_10` and `sensor_17` produced the highest segmentation complexity, suggesting frequent regime changes, whereas `sensor_33` and `sensor_28` appeared more stable.
- The clustering results show partial alignment with the RUL-based classes. Clusters 1 and 2 had high purity for Moderately Low RUL, while the largest cluster remained mixed, indicating that the unsupervised structure only partially matches the health classes.
- The Kadane analysis was especially informative, highlighting sensors such as `sensor_32`, `sensor_29`, `sensor_17`, and `sensor_37` as candidate degradation indicators because their strongest deviation intervals were concentrated in low-RUL regions.

### Challenges in This Project
- Selecting a variance threshold that is neither too coarse nor too fine.
- Splitting clusters without using external ML libraries.
- Working with a real dataset that may contain noisy or missing sensor readings.

### Limitations and Possible Improvements
- The segmentation threshold is global; adaptive thresholds may capture local dynamics better.
- The clustering rule uses Euclidean geometry only; normalization or robust distances may improve separation.
- The project brief mentions 52 sensors, but the execution file contained 50 available sensor columns, so analysis was limited to the columns actually present.
- The clustering results were imbalanced, with one dominant cluster containing most instances, which suggests that alternative split rules or feature scaling may yield more informative partitions.
