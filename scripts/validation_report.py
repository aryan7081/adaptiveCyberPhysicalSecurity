#!/usr/bin/env python3
"""
Technical validation: metrics, confusion matrix, ROC, noise robustness, trivial baselines.
Outputs figures/ and reports/validation_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import load_and_preprocess
from src.models.hybrid import HybridDetector
from src.torch_io import load_state_dict_checkpoint
from src.models.ocsvm import OCSVMDetector


def _mae_config_only(cfg: dict, num_features: int) -> dict:
    m = cfg["mae"]
    return {
        "hidden_dim": m["hidden_dim"],
        "num_layers": m["num_layers"],
        "num_heads": m["num_heads"],
        "dropout": m["dropout"],
        "mask_ratio": m["mask_ratio"],
        "init": m.get("init", "xavier"),
        "readout_mode": m.get("readout_mode", "mean_max"),
    }


def majority_baseline(y_true: np.ndarray) -> dict:
    """Always predict majority class (sanity check)."""
    maj = 0 if (y_true == 0).mean() >= 0.5 else 1
    pred = np.full_like(y_true, fill_value=maj)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    return {
        "name": "majority_class",
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def _random_baseline(y_true: np.ndarray, p_attack: float) -> dict:
    rng = np.random.RandomState(0)
    pred = (rng.rand(len(y_true)) < p_attack).astype(int)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    return {
        "name": f"random_attack_prob_{p_attack}",
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fast", action="store_true", help="5k sample subset")
    parser.add_argument("--noise-sigmas", default="0,0.05,0.1,0.2", help="Gaussian noise scale (× feature std)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    figures_dir = Path(cfg["paths"].get("figures_dir", "figures"))
    reports_dir = Path(cfg["paths"].get("reports_dir", "reports"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sample_size = 5000 if args.fast else 0
    _, _, X_test, y_test, X_benign, _ = load_and_preprocess(cfg, data_dir, sample_size=sample_size)
    num_features = X_benign.shape[1]

    import torch

    hybrid = HybridDetector(
        num_features=num_features,
        mae_config=_mae_config_only(cfg, num_features),
        ocsvm_config=cfg["ocsvm"],
        device=args.device,
    )
    mae_path = models_dir / "mae_pretrained.pt"
    if not mae_path.exists():
        mae_path = models_dir / "mae_best.pt"
    if not mae_path.exists():
        print("Train MAE first: python scripts/train_mae.py")
        return 1
    hybrid.mae.load_state_dict(load_state_dict_checkpoint(mae_path, args.device))
    hybrid.freeze_encoder()
    hybrid.fit_ocsvm(X_benign)

    metrics = hybrid.evaluate(X_test, y_test)
    y_pred = hybrid.predict(X_test)
    dec = hybrid.ocsvm.decision_function(hybrid.get_embeddings(X_test))

    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, -dec)
    roc_auc = auc(fpr, tpr)

    sigmas = [float(x) for x in args.noise_sigmas.split(",")]
    std = np.std(X_test, axis=0, keepdims=True) + 1e-8
    noise_results = {}
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(42)
    for s in sigmas:
        if s == 0:
            noise_results[str(s)] = {"f1": float(metrics["f1"]), "roc_auc": float(metrics["roc_auc"])}
            continue
        noise = rng.randn(*X_test.shape) * std * s
        Xn = X_test.astype(np.float64) + noise
        mn = hybrid.evaluate(Xn.astype(np.float32), y_test)
        noise_results[str(s)] = {"f1": float(mn["f1"]), "roc_auc": float(mn["roc_auc"])}

    baselines = {
        "majority": majority_baseline(y_test),
        "random_0.5": _random_baseline(y_test, 0.5),
    }

    out = {
        "hybrid_metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_auc_manual": float(roc_auc),
        "noise_robustness": noise_results,
        "baselines": baselines,
    }
    with open(reports_dir / "validation_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["normal", "anomaly"], cmap="Blues", ax=ax
        )
        plt.title("Hybrid (MAE + OCSVM) — confusion matrix")
        plt.tight_layout()
        plt.savefig(figures_dir / "confusion_matrix_hybrid.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        plt.title("Hybrid — ROC (score = −OCSVM decision function)")
        plt.tight_layout()
        plt.savefig(figures_dir / "roc_hybrid.png", dpi=150)
        plt.close()
        print(f"Saved figures to {figures_dir}/")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
