"""
Feature Engineering for Network Anomaly Detection
Domain-informed features: ratios, interactions, volatility-like statistics.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Feature names for reference
BASIC_RATIO_FEATURES = [
    "bytes_ratio",         # src_bytes / (dst_bytes + 1)
    "connection_intensity",  # count / (srv_count + 1)
]
INTERACTION_FEATURES = [
    "error_interaction",   # serror_rate * srv_serror_rate
    "host_service_density",  # dst_host_count * dst_host_srv_count (normalized)
]


class FeatureEngineer:
    """
    Create domain-informed features for network traffic.
    - Ratios (e.g., byte asymmetry)
    - Differences (e.g., rate deltas)
    - Interaction terms (e.g., error rate combinations)
    - Optional PCA for dimensionality reduction
    """

    def __init__(
        self,
        use_ratios: bool = True,
        use_interactions: bool = True,
        pca_components: int = 0,
        feature_names: Optional[List[str]] = None,
    ):
        self.use_ratios = use_ratios
        self.use_interactions = use_interactions
        self.pca_components = pca_components
        self.feature_names = feature_names or []
        self.pca_ = None
        self.scaler_ = StandardScaler()
        self._fitted = False

    def _get_column_index(self, name: str) -> int:
        """Get index of feature by name."""
        for i, n in enumerate(self.feature_names):
            if n == name:
                return i
        return -1

    def _resolve_indices(self) -> dict:
        """Resolve feature indices by name. Use standard NSL-KDD order if names available."""
        idx = self._get_column_index
        return {
            "src_bytes": idx("src_bytes"),
            "dst_bytes": idx("dst_bytes"),
            "count": idx("count"),
            "srv_count": idx("srv_count"),
            "serror_rate": idx("serror_rate"),
            "srv_serror_rate": idx("srv_serror_rate"),
            "dst_host_count": idx("dst_host_count"),
            "dst_host_srv_count": idx("dst_host_srv_count"),
        }

    def _create_derived(self, X: np.ndarray) -> np.ndarray:
        """Create ratio and interaction features from raw array."""
        extra = []
        m = self._resolve_indices()
        # Fallback: preprocessor order = numeric (excl. protocol,service,flag) + categorical
        src_b = m.get("src_bytes", 1)
        dst_b = m.get("dst_bytes", 2)
        count_i = m.get("count", 19)
        srv_count_i = m.get("srv_count", 20)
        serror_i = m.get("serror_rate", 21)
        srv_serror_i = m.get("srv_serror_rate", 22)
        host_count_i = m.get("dst_host_count", 28)
        host_srv_i = m.get("dst_host_srv_count", 29)

        if self.use_ratios:
            sb = X[:, src_b]
            db = X[:, dst_b]
            extra.append((sb / (db + 1e-6)).reshape(-1, 1))
            cnt = X[:, count_i]
            srv = X[:, srv_count_i]
            extra.append((cnt / (srv + 1e-6)).reshape(-1, 1))
        if self.use_interactions:
            e1 = X[:, serror_i]
            e2 = X[:, srv_serror_i]
            extra.append((e1 * e2).reshape(-1, 1))
            hc = X[:, host_count_i]
            hs = X[:, host_srv_i]
            extra.append((hc * hs / (1 + hc + hs)).reshape(-1, 1))
        if extra:
            return np.hstack([X] + extra)
        return X

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "FeatureEngineer":
        """Fit engineer on training data."""
        if feature_names:
            self.feature_names = feature_names
        X_derived = self._create_derived(X)
        self.scaler_.fit(X_derived)
        X_scaled = self.scaler_.transform(X_derived)
        if self.pca_components > 0:
            self.pca_ = PCA(n_components=self.pca_components, random_state=42)
            self.pca_.fit(X_scaled)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data with derived features and optional PCA."""
        X_derived = self._create_derived(X)
        X_scaled = self.scaler_.transform(X_derived)
        if self.pca_ is not None:
            X_scaled = self.pca_.transform(X_scaled)
        return X_scaled.astype(np.float32)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, feature_names)
        return self.transform(X)
