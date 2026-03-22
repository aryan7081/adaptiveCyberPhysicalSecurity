#!/usr/bin/env python3
"""
Pre-train MAE on benign traffic only.
Usage: python scripts/train_mae.py [--config config/config.yaml]
"""

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import NSLKDDLoader
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.mae import TabularMAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load benign-only training data
    loader = NSLKDDLoader(str(data_dir))
    train_benign, _ = loader.load_benign_only(
        benign_label=cfg["dataset"]["benign_label"]
    )

    # Preprocess
    preproc = DataPreprocessor(
        categorical_cols=cfg["features"]["categorical"],
        log_transform_cols=cfg["features"].get("log_transform", []),
    )
    X, _ = preproc.fit_transform(train_benign, include_label=False)
    feature_names = preproc.feature_names_

    # Feature engineering (optional)
    fe_config = cfg["features"]
    feat_eng = FeatureEngineer(
        use_ratios=True,
        use_interactions=True,
        pca_components=fe_config.get("pca_components", 0),
    )
    X = feat_eng.fit_transform(X, feature_names)
    num_features = X.shape[1]

    # MAE config
    mae_cfg = cfg["mae"]
    mae = TabularMAE(
        num_features=num_features,
        hidden_dim=mae_cfg["hidden_dim"],
        num_layers=mae_cfg["num_layers"],
        num_heads=mae_cfg["num_heads"],
        dropout=mae_cfg["dropout"],
        mask_ratio=mae_cfg["mask_ratio"],
        init=mae_cfg.get("init", "xavier"),
    ).to(args.device)
    opt = torch.optim.AdamW(
        mae.parameters(),
        lr=float(mae_cfg["lr"]),
        weight_decay=float(mae_cfg.get("weight_decay", 1e-5)),
    )
    epochs = args.epochs or mae_cfg["epochs"]
    batch_size = args.batch_size or mae_cfg["batch_size"]

    # DataLoader
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop with early stopping
    best_loss = float("inf")
    patience = 10
    no_improve = 0
    for ep in range(epochs):
        mae.train()
        total_loss = 0
        for (batch,) in tqdm(loader_dl, desc=f"Epoch {ep+1}/{epochs}", leave=False):
            batch = batch.to(args.device)
            loss, _, _ = mae(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader_dl)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(mae.state_dict(), models_dir / "mae_best.pt")
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break
        print(f"Epoch {ep+1} Loss: {avg_loss:.4f} Best: {best_loss:.4f}")

    # Save final
    mae.load_state_dict(torch.load(models_dir / "mae_best.pt"))
    torch.save(mae.state_dict(), models_dir / "mae_pretrained.pt")
    print(f"Saved MAE to {models_dir / 'mae_pretrained.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
