# Water Pump RUL - Final Working Deliverable

This submission is a complete from-scratch algorithmic project for the Water Pump RUL dataset.

## What is included
- Divide-and-Conquer segmentation for 10 selected sensors
- Recursive top-down clustering for 4 clusters using all sensor features
- Kadane-based maximum deviation interval analysis for all sensors
- Toy examples for correctness verification
- Auto-generated report after execution
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
