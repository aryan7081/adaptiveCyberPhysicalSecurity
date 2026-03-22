#!/usr/bin/env python3
"""
Quick pipeline test: Model A (OCSVM only) without PyTorch.
Run this if PyTorch is not installed to verify data loading and preprocessing.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import NSLKDDLoader
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.ocsvm import OCSVMDetector


def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    loader = NSLKDDLoader(str(data_dir))
    train_df, test_df = loader.load()

    preproc = DataPreprocessor(
        categorical_cols=cfg["features"]["categorical"],
        log_transform_cols=cfg["features"].get("log_transform", []),
    )
    X_train, y_train = preproc.fit_transform(train_df)
    X_test, y_test = preproc.transform(test_df, include_label=True)

    feat_eng = FeatureEngineer(use_ratios=True, use_interactions=True)
    X_train = feat_eng.fit_transform(X_train, preproc.feature_names_)
    X_test = feat_eng.transform(X_test)

    train_benign = train_df[train_df["label"].astype(str).str.lower() == "normal"]
    X_benign, _ = preproc.transform(train_benign, include_label=False)
    X_benign = feat_eng.transform(X_benign)

    print(f"X_benign: {X_benign.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    oc = OCSVMDetector(kernel="rbf", nu=0.01, gamma="scale")
    oc.fit(X_benign)
    metrics = oc.evaluate(X_test, y_test)
    print("Model A (OCSVM) metrics:", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
