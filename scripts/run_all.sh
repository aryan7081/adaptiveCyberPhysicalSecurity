#!/bin/bash
# Run full Phase 1 pipeline
set -e
cd "$(dirname "$0")/.."

echo "=== Phase 1: Adaptive CPS Pipeline ==="

echo "[1/5] Downloading NSL-KDD..."
python scripts/download_data.py

echo "[2/5] Running EDA..."
python scripts/run_eda.py

echo "[3/5] Generating architecture diagram..."
python figures/architecture_diagram.py

echo "[4/5] Pre-training MAE..."
python scripts/train_mae.py

echo "[5/5] Running ablation study..."
python scripts/run_ablation.py

echo "=== Done. Check results/ablation_table.csv and figures/ ==="
