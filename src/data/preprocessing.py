"""
Data Preprocessing for NSL-KDD
Handles missing values, type coercion, duplicates, and basic cleaning.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

LABEL_COL = "label"


class DataPreprocessor:
    """
    Preprocess NSL-KDD for modeling.
    - Fix dtype issues (strings in float columns)
    - Handle missing/NaN
    - Encode categoricals
    - Scale numerics
    """

    def __init__(
        self,
        categorical_cols: List[str],
        log_transform_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
    ):
        self.categorical_cols = [c for c in categorical_cols if c != LABEL_COL]
        self.log_transform_cols = log_transform_cols or []
        self.exclude_cols = exclude_cols or []
        self.label_encoders_ = {}
        self.scaler_ = StandardScaler()
        self.imputer_ = SimpleImputer(strategy="median")
        self.feature_names_: List[str] = []

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce numeric columns; replace non-numeric with NaN then impute."""
        numeric_cols = [
            c for c in df.columns
            if c not in self.categorical_cols and c != LABEL_COL and c not in self.exclude_cols
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _log_transform(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Log1p transform for heavy-tailed features."""
        for col in self.log_transform_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                vals = np.clip(vals, 0, None)  # Ensure non-negative
                df[col] = np.log1p(vals)
        return df

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit preprocessor on training data."""
        df = df.copy()
        df = self._coerce_numeric(df)
        df = self._log_transform(df, fit=True)

        numeric_cols = [
            c for c in df.columns
            if c not in self.categorical_cols and c != LABEL_COL and c not in self.exclude_cols
        ]
        if numeric_cols:
            X_num = df[numeric_cols]
            self.imputer_.fit(X_num)
            X_imp = self.imputer_.transform(X_num)
            self.scaler_.fit(X_imp)

        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        for col in cat_cols:
            le = LabelEncoder()
            vals = df[col].astype(str).fillna("__MISSING__").tolist()
            le.fit(vals + ["__UNK__"])  # Add UNK for unseen categories at transform
            self.label_encoders_[col] = le
        self.feature_names_ = numeric_cols + cat_cols
        return self

    def transform(
        self,
        df: pd.DataFrame,
        include_label: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform dataframe to numeric array.
        Returns (X, y) where y is None if no label column.
        """
        df = df.copy()
        df = self._coerce_numeric(df)
        df = self._log_transform(df)

        numeric_cols = [
            c for c in df.columns
            if c not in self.categorical_cols and c != LABEL_COL and c not in self.exclude_cols
        ]
        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        cols = numeric_cols + cat_cols

        parts = []
        if numeric_cols:
            X_num = self.imputer_.transform(df[numeric_cols])
            X_num = self.scaler_.transform(X_num)
            parts.append(X_num)
        for col in cat_cols:
            le = self.label_encoders_[col]
            encoded = df[col].astype(str).fillna("__MISSING__")
            # Handle unseen categories (e.g. new services in test set)
            encoded = encoded.apply(lambda v: v if v in le.classes_ else "__UNK__")
            # Ensure all are in classes (le was fit with __UNK__)
            encoded = le.transform(encoded)
            parts.append(encoded.reshape(-1, 1))
        X = np.hstack(parts).astype(np.float32)

        y = None
        if include_label and LABEL_COL in df.columns:
            y = (df[LABEL_COL].astype(str).str.lower() != "normal").astype(int).values
        return X, y

    def fit_transform(
        self,
        df: pd.DataFrame,
        include_label: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df, include_label)
