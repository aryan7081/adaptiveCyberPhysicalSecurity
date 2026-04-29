"""Dataset loaders for NSL-KDD and CIC-IDS."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# NSL-KDD 41 feature names (standard schema)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
LABEL_COL = "label"
DIFFICULTY_COL = "difficulty"  # Present in test set only


class NSLKDDLoader:
    """
    Load and validate NSL-KDD dataset.
    Supports .txt (CSV-like) and .csv formats.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        train_file: str = "KDDTrain+.txt",
        test_file: str = "KDDTest+.txt",
        sep: str = ",",
        label_col: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test splits.
        Returns (train_df, test_df) with columns properly named.
        """
        train_path = self.data_dir / train_file
        test_path = self.data_dir / test_file

        if not train_path.exists():
            raise FileNotFoundError(
                f"Training file not found: {train_path}\n"
                "Download from: https://github.com/defcom17/NSL_KDD\n"
                "Place KDDTrain+.txt and KDDTest+.txt in data/raw/"
            )
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test file not found: {test_path}\n"
                "Download from: https://github.com/defcom17/NSL_KDD"
            )

        # Read with 43 column names; NSL-KDD has 41 features + label + difficulty
        all_cols = NSL_KDD_COLUMNS + [LABEL_COL, DIFFICULTY_COL]
        train_df = pd.read_csv(train_path, names=all_cols, sep=sep, header=None)
        test_df = pd.read_csv(test_path, names=all_cols, sep=sep, header=None)
        # If only 42 cols present, pandas will have NaN for difficulty
        if train_df.shape[1] < 43:
            pass  # already 42 cols
        else:
            train_df = train_df[NSL_KDD_COLUMNS + [LABEL_COL]]
            test_df = test_df[NSL_KDD_COLUMNS + [LABEL_COL]]
        return train_df, test_df

    def load_benign_only(
        self,
        train_file: str = "KDDTrain+.txt",
        test_file: Optional[str] = None,
        benign_label: str = "normal",
        benign_labels: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        sep: str = ",",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load only benign (normal) traffic for unsupervised pre-training.
        Optionally also return full test set for evaluation.
        """
        train_df, test_df = self.load(train_file, test_file or "KDDTest+.txt", sep=sep)
        benign_labels = benign_labels or [benign_label]
        benign_set = {str(v).strip().lower() for v in benign_labels}
        train_benign = train_df[train_df[LABEL_COL].astype(str).str.strip().str.lower().isin(benign_set)].copy()
        train_benign = train_benign.reset_index(drop=True)

        if test_file:
            return train_benign, test_df
        return train_benign, None

    @staticmethod
    def download_from_github(data_dir: str = "data/raw") -> None:
        """
        Attempt to download NSL-KDD from a mirror.
        User may need to download manually if blocked.
        """
        import urllib.request

        base = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        for fname in ["KDDTrain+.txt", "KDDTest+.txt"]:
            url = f"{base}/{fname}"
            dest = data_path / fname
            if not dest.exists():
                try:
                    urllib.request.urlretrieve(url, dest)
                    print(f"Downloaded {fname}")
                except Exception as e:
                    print(f"Could not download {fname}: {e}")
                    print(f"Manual: {url}")


class CICIDSLoader:
    """Load CIC-IDS CSV files with BENIGN vs attack labels."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @staticmethod
    def _find_label_col(df: pd.DataFrame, configured_label_col: Optional[str] = None) -> str:
        if configured_label_col and configured_label_col in df.columns:
            return configured_label_col
        for candidate in ["Label", "label", " Label"]:
            if candidate in df.columns:
                return candidate
        raise ValueError(
            "Could not find label column in CIC-IDS dataframe. "
            "Set dataset.label_col in config to match your CSV schema."
        )

    @staticmethod
    def _clean_numeric(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if col == label_col:
                continue
            if out[col].dtype == object:
                out[col] = out[col].astype(str).str.strip()
            out[col] = pd.to_numeric(out[col], errors="ignore")
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def load(
        self,
        train_file: str,
        test_file: Optional[str] = None,
        label_col: Optional[str] = None,
        sep: str = ",",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_path = self.data_dir / train_file
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training file not found: {train_path}\n"
                "Download/extract CIC-IDS CSVs and place them in data/raw/."
            )
        train_df = pd.read_csv(train_path, sep=sep, low_memory=False)
        train_df = self._normalize_columns(train_df)
        label_col = self._find_label_col(train_df, label_col)
        train_df = self._clean_numeric(train_df, label_col)
        train_df = train_df.rename(columns={label_col: LABEL_COL})

        if test_file:
            test_path = self.data_dir / test_file
            if not test_path.exists():
                raise FileNotFoundError(f"Test file not found: {test_path}")
            test_df = pd.read_csv(test_path, sep=sep, low_memory=False)
            test_df = self._normalize_columns(test_df)
            test_label_col = self._find_label_col(test_df, label_col)
            test_df = self._clean_numeric(test_df, test_label_col)
            test_df = test_df.rename(columns={test_label_col: LABEL_COL})
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

        # If explicit test split is unavailable, create deterministic split.
        rng = np.random.RandomState(random_state)
        idx = np.arange(len(train_df))
        rng.shuffle(idx)
        cut = int(len(train_df) * (1.0 - test_size))
        train_idx, test_idx = idx[:cut], idx[cut:]
        split_train = train_df.iloc[train_idx].reset_index(drop=True)
        split_test = train_df.iloc[test_idx].reset_index(drop=True)
        return split_train, split_test

    def load_benign_only(
        self,
        train_file: str,
        test_file: Optional[str] = None,
        benign_labels: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        sep: str = ",",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        benign_labels = benign_labels or ["BENIGN"]
        benign_set = {str(v).strip().lower() for v in benign_labels}
        train_df, test_df = self.load(
            train_file=train_file,
            test_file=test_file,
            label_col=label_col,
            sep=sep,
            test_size=test_size,
            random_state=random_state,
        )
        train_benign = train_df[train_df[LABEL_COL].astype(str).str.strip().str.lower().isin(benign_set)].copy()
        train_benign = train_benign.reset_index(drop=True)
        return train_benign, test_df


def create_loader(dataset_cfg: dict, data_dir: str):
    """Return dataset-specific loader from config."""
    dataset_name = str(dataset_cfg.get("name", "nsl_kdd")).lower()
    if dataset_name in {"nsl_kdd", "nsl-kdd", "nsl"}:
        return NSLKDDLoader(data_dir)
    if dataset_name in {"cic_ids", "cic-ids", "cicids2017", "cic_ids_2017"}:
        return CICIDSLoader(data_dir)
    raise ValueError(f"Unsupported dataset.name: {dataset_cfg.get('name')}")
