#!/bin/bash
# Run full Phase 1 pipeline
set -e
cd "$(dirname "$0")/.."

echo "=== Phase 1: Adaptive CPS Pipeline ==="

echo "[1/5] Preparing dataset..."
python scripts/download_data.py

echo "[2/5] Running EDA..."
python scripts/run_eda.py

echo "[3/5] Generating architecture diagram..."
python figures/architecture_diagram.py

echo "[4/5] Pre-training MAE..."
python scripts/train_mae.py

echo "[5/7] Running ablation study..."
python scripts/run_ablation.py

echo "[6/7] Validation report (confusion matrix, ROC, robustness)..."
python scripts/validation_report.py --fast

echo "[7/7] Optional: SHAP plots (install shap)..."
python scripts/explainability.py --mode raw --fast || true

echo "=== Done. Check results/, reports/, figures/ ==="
