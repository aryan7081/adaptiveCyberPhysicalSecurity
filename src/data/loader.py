"""
NSL-KDD Data Loader
Handles downloading, loading, and basic validation of NSL-KDD dataset.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

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
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load only benign (normal) traffic for unsupervised pre-training.
        Optionally also return full test set for evaluation.
        """
        train_df, test_df = self.load(train_file, test_file or "KDDTest+.txt")

        train_benign = train_df[train_df[LABEL_COL].astype(str).str.lower() == benign_label.lower()].copy()
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
