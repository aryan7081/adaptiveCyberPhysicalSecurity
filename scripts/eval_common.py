"""Shared load + preprocess for evaluation scripts."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.data.loader import create_loader
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer

ENG_EXTRA_NAMES = [
    "bytes_ratio",
    "connection_intensity",
    "error_interaction",
    "host_service_density",
]


def load_and_preprocess(
    cfg: Dict[str, Any],
    data_dir: Path,
    sample_size: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    ds = cfg["dataset"]
    loader = create_loader(ds, str(data_dir))
    train_df, test_df = loader.load(
        train_file=ds["train_file"],
        test_file=ds.get("test_file"),
        sep=ds.get("sep", ","),
        label_col=ds.get("label_col"),
        test_size=float(ds.get("test_size", 0.2)),
        random_state=int(cfg["project"].get("seed", 42)),
    )
    if sample_size > 0:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
    preproc = DataPreprocessor(
        categorical_cols=cfg["features"]["categorical"],
        log_transform_cols=cfg["features"].get("log_transform", []),
        exclude_cols=cfg["features"].get("exclude", []),
        benign_labels=cfg["dataset"].get("benign_labels", [cfg["dataset"].get("benign_label", "normal")]),
    )
    X_train, y_train = preproc.fit_transform(train_df)
    X_test, y_test = preproc.transform(test_df, include_label=True)
    feat_eng = FeatureEngineer(
        use_ratios=True,
        use_interactions=True,
        pca_components=cfg["features"].get("pca_components", 0),
    )
    X_train = feat_eng.fit_transform(X_train, preproc.feature_names_)
    X_test = feat_eng.transform(X_test)
    benign_vals = {str(v).strip().lower() for v in cfg["dataset"].get("benign_labels", [cfg["dataset"].get("benign_label", "normal")])}
    train_benign = train_df[train_df["label"].astype(str).str.strip().str.lower().isin(benign_vals)]
    X_benign, _ = preproc.transform(train_benign, include_label=False)
    X_benign = feat_eng.transform(X_benign)

    names = list(preproc.feature_names_)
    if feat_eng.use_ratios:
        names.extend(ENG_EXTRA_NAMES[:2])
    if feat_eng.use_interactions:
        names.extend(ENG_EXTRA_NAMES[2:])
    if len(names) < X_test.shape[1]:
        for j in range(len(names), X_test.shape[1]):
            names.append(f"feature_{j}")
    names = names[: X_test.shape[1]]

    return X_train, y_train, X_test, y_test, X_benign, names
