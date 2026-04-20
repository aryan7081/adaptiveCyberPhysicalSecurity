#!/usr/bin/env python3
"""
SHAP analysis for One-Class SVM decisions (feature-level or embedding-level).
Install: pip install shap matplotlib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import load_and_preprocess
from src.models.hybrid import HybridDetector
from src.torch_io import load_state_dict_checkpoint
from src.models.ocsvm import OCSVMDetector, parse_ocsvm_section


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--mode",
        choices=["raw", "embedding"],
        default="raw",
        help="raw: OCSVM on preprocessed features; embedding: OCSVM on MAE embeddings",
    )
    parser.add_argument("--max-background", type=int, default=300)
    parser.add_argument("--max-explain", type=int, default=80)
    parser.add_argument(
        "--max-evals",
        type=int,
        default=2500,
        help="SHAP permutation budget (lower = faster, noisier). Try 800–4000.",
    )
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install SHAP: pip install shap matplotlib")
        return 1

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    figures_dir = Path(cfg["paths"].get("figures_dir", "figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    sample_size = 5000 if args.fast else 0
    _, _, X_test, _y_test, X_benign, feat_names = load_and_preprocess(
        cfg, data_dir, sample_size=sample_size
    )
    num_features = X_benign.shape[1]

    if args.fast:
        args.max_background = min(args.max_background, 200)
        args.max_explain = min(args.max_explain, 40)
        args.max_evals = min(args.max_evals, 1200)

    import torch

    if args.mode == "raw":
        svm_kw, _, _ = parse_ocsvm_section(cfg["ocsvm"])
        oc = OCSVMDetector(**svm_kw)
        oc.fit(X_benign)
        clf = oc.clf
        bg = X_benign[: args.max_background].astype(np.float64)
        ex = X_test[: args.max_explain].astype(np.float64)
        names = feat_names
    else:
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
            print("Train MAE first for embedding mode.")
            return 1
        hybrid.mae.load_state_dict(load_state_dict_checkpoint(mae_path, args.device))
        hybrid.freeze_encoder()
        hybrid.fit_ocsvm(X_benign)
        clf = hybrid.ocsvm.clf
        bg = hybrid.get_embeddings(X_benign[: args.max_background])
        ex = hybrid.get_embeddings(X_test[: args.max_explain])
        h = bg.shape[1]
        names = [f"emb_{i}" for i in range(h)]

    def model_score(X):
        """Higher = more anomalous (aligns with y=1)."""
        return -clf.decision_function(X)

    print(
        "SHAP: building explainer (often uses PermutationExplainer for kernel SVM). "
        "This can take several minutes on CPU; reduce --max-explain / --max-evals for speed.",
        flush=True,
    )
    print(
        f"  background={len(bg)} rows, explain={len(ex)} rows, max_evals={args.max_evals}",
        flush=True,
    )
    explainer = shap.Explainer(model_score, bg, feature_names=names)
    print("SHAP: computing values…", flush=True)
    try:
        sv = explainer(ex, max_evals=args.max_evals)
    except TypeError:
        sv = explainer(ex)
    print("SHAP: done.", flush=True)

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(sv, max_display=20, show=False)
    out = figures_dir / f"shap_beeswarm_{args.mode}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # Quick check: mean |SHAP| ranking
    vals = np.abs(sv.values).mean(axis=0)
    order = np.argsort(-vals)[:15]
    print("Top dimensions by mean |SHAP|:")
    for i in order:
        print(f"  {names[i]}: {vals[i]:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
