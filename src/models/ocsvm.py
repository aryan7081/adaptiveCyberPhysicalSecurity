"""
One-Class SVM for Anomaly Detection
Uses embeddings from frozen MAE encoder to define decision boundaries.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib


class OCSVMDetector:
    """
    One-Class SVM trained on benign (normal) embeddings.
    Anomalies fall outside the learned decision boundary.
    """

    def __init__(self, kernel: str = "rbf", nu: float = 0.01, gamma: str = "scale"):
        self.clf = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self._fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OCSVMDetector":
        """
        Fit on benign samples only.
        X: embeddings from MAE encoder (or raw features for baseline).
        """
        self.clf.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns +1 for inliers (normal), -1 for outliers (anomaly)."""
        pred = self.clf.predict(X)
        return pred

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to decision boundary. Negative = anomaly."""
        return self.clf.decision_function(X)

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Map SVM output to 0 (normal) / 1 (anomaly)."""
        pred = self.predict(X)
        return (pred == -1).astype(int)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """Compute metrics. y_true: 0=normal, 1=anomaly."""
        y_pred = self.predict_binary(X)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            dec = self.decision_function(X)
            auc = roc_auc_score(y_true, -dec)  # flip: higher = more normal
        except Exception:
            auc = 0.0
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "OCSVMDetector":
        return joblib.load(path)
