#!/usr/bin/env python3
"""
Pre-train MAE on benign traffic only.
Validation split for early stopping; logs learning curves; optional Mixup and LR schedule.
Usage: python scripts/train_mae.py [--config config/config.yaml]
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import NSLKDDLoader
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.mae import TabularMAE
from src.torch_io import load_state_dict_checkpoint


def mixup_tensor(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Tabular Mixup for self-supervised reconstruction (train only)."""
    if alpha <= 0:
        return x
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx]


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
    reports_dir = Path(cfg["paths"].get("reports_dir", "reports"))
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    loader = NSLKDDLoader(str(data_dir))
    train_benign, _ = loader.load_benign_only(
        benign_label=cfg["dataset"]["benign_label"]
    )

    preproc = DataPreprocessor(
        categorical_cols=cfg["features"]["categorical"],
        log_transform_cols=cfg["features"].get("log_transform", []),
    )
    X, _ = preproc.fit_transform(train_benign, include_label=False)
    feature_names = preproc.feature_names_

    fe_config = cfg["features"]
    feat_eng = FeatureEngineer(
        use_ratios=True,
        use_interactions=True,
        pca_components=fe_config.get("pca_components", 0),
    )
    X = feat_eng.fit_transform(X, feature_names)
    num_features = X.shape[1]

    mae_cfg = cfg["mae"]
    val_ratio = float(mae_cfg.get("val_ratio", 0.15))
    n = len(X)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train = X[train_idx].astype(np.float32)
    X_val = X[val_idx].astype(np.float32)

    mae = TabularMAE(
        num_features=num_features,
        hidden_dim=mae_cfg["hidden_dim"],
        num_layers=mae_cfg["num_layers"],
        num_heads=mae_cfg["num_heads"],
        dropout=mae_cfg["dropout"],
        mask_ratio=mae_cfg["mask_ratio"],
        init=mae_cfg.get("init", "xavier"),
        readout_mode=mae_cfg.get("readout_mode", "mean_max"),
    ).to(args.device)
    opt = torch.optim.AdamW(
        mae.parameters(),
        lr=float(mae_cfg["lr"]),
        weight_decay=float(mae_cfg.get("weight_decay", 1e-5)),
    )
    epochs = args.epochs or mae_cfg["epochs"]
    batch_size = args.batch_size or mae_cfg["batch_size"]
    mixup_alpha = float(mae_cfg.get("mixup_alpha", 0.0))
    sched_name = str(mae_cfg.get("lr_scheduler", "cosine")).lower()
    patience = int(mae_cfg.get("early_stopping_patience", 12))

    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-6)
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=4, min_lr=1e-6
        )
    else:
        scheduler = None

    train_ds = TensorDataset(torch.from_numpy(X_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    history_path = reports_dir / "mae_training_history.csv"
    best_val = float("inf")
    no_improve = 0
    history_rows = []

    for ep in range(epochs):
        mae.train()
        total_loss = 0.0
        for (batch,) in tqdm(train_loader, desc=f"Epoch {ep + 1}/{epochs}", leave=False):
            batch = batch.to(args.device)
            if mixup_alpha > 0 and batch.size(0) > 1:
                batch = mixup_tensor(batch, mixup_alpha)
            loss, _, _ = mae(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        train_loss = total_loss / max(1, len(train_loader))

        mae.eval()
        val_chunks = []
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                vb = torch.from_numpy(X_val[i : i + batch_size]).to(args.device)
                vloss, _, _ = mae(vb)
                val_chunks.append(vloss.item())
        val_loss = float(np.mean(val_chunks)) if val_chunks else 0.0

        lr_now = opt.param_groups[0]["lr"]
        history_rows.append(
            {"epoch": ep + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now}
        )

        if scheduler is not None:
            if sched_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(mae.state_dict(), models_dir / "mae_best.pt")
        else:
            no_improve += 1

        print(
            f"Epoch {ep + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_val={best_val:.6f} lr={lr_now:.2e}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {ep + 1} (no val improvement for {patience} epochs)")
            break

    with open(history_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        w.writeheader()
        w.writerows(history_rows)
    print(f"Saved training history to {history_path}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        epnums = [r["epoch"] for r in history_rows]
        ax.plot(epnums, [r["train_loss"] for r in history_rows], label="train")
        ax.plot(epnums, [r["val_loss"] for r in history_rows], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE reconstruction loss")
        ax.legend()
        ax.set_title("MAE learning curves")
        fig.tight_layout()
        fig_path = Path(cfg["paths"].get("figures_dir", "figures")) / "mae_learning_curves.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved {fig_path}")
    except ImportError:
        pass

    mae.load_state_dict(load_state_dict_checkpoint(models_dir / "mae_best.pt", args.device))
    torch.save(mae.state_dict(), models_dir / "mae_pretrained.pt")
    print(f"Saved MAE to {models_dir / 'mae_pretrained.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
