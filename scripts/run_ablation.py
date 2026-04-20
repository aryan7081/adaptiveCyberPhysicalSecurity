#!/usr/bin/env python3
"""
Ablation Study: Model A (Adv ML only), Model B (DL only), Model C (Hybrid).
Produces results table for the report.
"""

import argparse
import sys
import warnings
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import load_and_preprocess
from src.models.mae import TabularMAE
from src.torch_io import load_state_dict_checkpoint
from src.models.ocsvm import OCSVMDetector, parse_ocsvm_section
from src.models.hybrid import HybridDetector

warnings.filterwarnings(
    "ignore",
    message=".*enable_nested_tensor.*",
    module="torch.nn.modules.transformer",
)


def _mae_kwargs(cfg: dict, num_features: int) -> dict:
    m = cfg["mae"]
    return {
        "num_features": num_features,
        "hidden_dim": m["hidden_dim"],
        "num_layers": m["num_layers"],
        "num_heads": m["num_heads"],
        "dropout": m["dropout"],
        "mask_ratio": m["mask_ratio"],
        "init": m.get("init", "xavier"),
        "readout_mode": m.get("readout_mode", "mean_max"),
    }


def run_model_a(X_benign: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, cfg: dict) -> dict:
    """Model A: One-Class SVM on raw (preprocessed) features only - Adv ML only."""
    svm_kw, _, _ = parse_ocsvm_section(cfg["ocsvm"])
    oc = OCSVMDetector(**svm_kw)
    oc.fit(X_benign)
    return oc.evaluate(X_test, y_test)


def run_model_b(X_benign: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, cfg: dict, device: str, models_dir: Path, batch_size: int = 2048) -> dict:
    """Model B: MAE reconstruction error as anomaly score - DL only."""
    num_features = X_benign.shape[1]
    mae_cfg = cfg["mae"]
    mae = TabularMAE(
        **_mae_kwargs(cfg, num_features),
    ).to(device)
    mae_path = models_dir / "mae_pretrained.pt"
    if not mae_path.exists():
        mae_path = models_dir / "mae_best.pt"
    if mae_path.exists():
        mae.load_state_dict(load_state_dict_checkpoint(mae_path, device))
    else:
        print("Warning: No pretrained MAE. Skipping Model B.")
        return {}
    mae.eval()

    def _recon_loss(X: np.ndarray) -> np.ndarray:
        losses = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            t = torch.from_numpy(batch.astype(np.float32)).to(device)
            with torch.no_grad():
                _, _, recon = mae(t, mask=None, no_mask=True)
                l = torch.mean((recon - t) ** 2, dim=1).cpu().numpy()
            losses.append(l)
        return np.concatenate(losses)

    recon_loss = _recon_loss(X_test)
    loss_b = _recon_loss(X_benign)
    thresh = np.median(loss_b) + 2 * np.std(loss_b)
    y_pred = (recon_loss > thresh).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, recon_loss) if len(np.unique(y_test)) > 1 else 0,
    }


def run_model_c(X_benign: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, cfg: dict, device: str, models_dir: Path) -> dict:
    """Model C: Hybrid - MAE (frozen) + One-Class SVM on embeddings."""
    num_features = X_benign.shape[1]
    mk = _mae_kwargs(cfg, num_features)
    del mk["num_features"]
    hybrid = HybridDetector(
        num_features=num_features,
        mae_config=mk,
        ocsvm_config=cfg["ocsvm"],
        device=device,
    )
    mae_path = models_dir / "mae_pretrained.pt"
    if not mae_path.exists():
        mae_path = models_dir / "mae_best.pt"
    if mae_path.exists():
        hybrid.mae.load_state_dict(load_state_dict_checkpoint(mae_path, device))
    hybrid.freeze_encoder()
    hybrid.fit_ocsvm(X_benign)
    return hybrid.evaluate(X_test, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/ablation_table.csv")
    parser.add_argument("--fast", action="store_true", help="Use 5k samples to avoid OOM (~2-3 min)")
    parser.add_argument("--batch-size", type=int, default=1024, help="MAE inference batch size (smaller = less RAM)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    sample_size = 5000 if args.fast else 0
    if args.fast:
        print("FAST MODE: 5k samples (use to avoid OOM on limited RAM)")
    print("Loading and preprocessing data...")
    X_train, _y_train, X_test, y_test, X_benign, _ = load_and_preprocess(
        cfg, data_dir, sample_size=sample_size
    )
    print(f"Train: {X_train.shape}, Benign: {X_benign.shape}, Test: {X_test.shape}")

    results = {}

    print("Model A: Advanced ML only (One-Class SVM on raw features)...")
    results["Model_A_AdvML_Only"] = run_model_a(X_benign, X_test, y_test, cfg)
    print(results["Model_A_AdvML_Only"])

    print("Model B: Deep Learning only (MAE reconstruction threshold)...")
    results["Model_B_DL_Only"] = run_model_b(X_benign, X_test, y_test, cfg, args.device, models_dir, batch_size=args.batch_size)
    if results["Model_B_DL_Only"]:
        print(results["Model_B_DL_Only"])
    else:
        print("Skipped (no pretrained MAE)")

    print("Model C: Hybrid (MAE + OCSVM)...")
    results["Model_C_Hybrid"] = run_model_c(X_benign, X_test, y_test, cfg, args.device, models_dir)
    print(results["Model_C_Hybrid"])

    # Build table
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, r in results.items():
        if not r:
            continue
        row = {"Model": name}
        for m in metrics:
            row[m] = round(r.get(m, 0), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nAblation table saved to {args.output}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
