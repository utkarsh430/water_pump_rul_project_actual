# Project: Algorithmic Time‑Series Segmentation and Condition Analysis Using Water Pump RUL Dataset

Objective is to develop an algorithm‑driven system to segment, cluster, and analyze time‑series sensor data from a real water‑pump machine.
The system will use core algorithmic techniques—Divide‑and‑Conquer segmentation, Closest Pair, and Maximum Subarray—to characterize machine behavior and relate the results to four machine‑health categories derived from RUL (Remaining Useful Life).

## Important Files

- `main.py` is the main entry point of the project and contains the complete execution flow for all tasks.
- `report.md` is the final written report for the project and includes the methodology, toy-example verification, dataset description, execution results, discussion, and conclusions.

## Dataset

This project uses the Water Pump RUL dataset downloaded from Kaggle.

- Source: https://www.kaggle.com/datasets/anseldsouza/water-pump-rul-predictive-maintenance
- File used for execution: `rul_hrs.csv`
  
## What is included
- Divide-and-Conquer segmentation for 10 selected sensors
- Recursive top-down clustering for 4 clusters using all sensor features
- Kadane-based maximum deviation interval analysis for all sensors
- Toy examples for correctness verification
- CSV summaries, JSON summary, and PNG plots

## Folder structure
- `main.py` - run everything
- `src/` - modular source code
- `requirements.txt` - dependencies
- `output/` - created after running the project

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py --csv_path "C:\path\to\your\water_pump_dataset.csv"
```

## Expected dataset format
The CSV must contain:
- `timestamp`
- `rul`
- sensor columns named like `sensor_00`, `sensor_01`, ..., `sensor_51`



## Generated deliverables after execution
- `output/task1_segmentation_summary.csv`
- `output/task2_clustering_summary.csv`
- `output/task2_cluster_assignments.csv`
- `output/task3_kadane_summary.csv`
- `output/summary.json`
- `output/report.md`
- segmentation and kadane plots in output subfolders
