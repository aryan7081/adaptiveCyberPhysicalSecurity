"""
Hybrid Detector: MAE (frozen) + One-Class SVM
Combines learned representations with explicit anomaly boundaries.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

from .mae import TabularMAE
from .ocsvm import OCSVMDetector


class HybridDetector:
    """
    Phase 1: Pre-train MAE on benign traffic.
    Phase 2: Freeze MAE encoder, extract embeddings.
    Phase 3: Train One-Class SVM on embeddings.
    Inference: MAE embedding -> OCSVM decision.
    """

    def __init__(
        self,
        num_features: int,
        mae_config: Optional[Dict[str, Any]] = None,
        ocsvm_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        mae_config = mae_config or {}
        ocsvm_config = ocsvm_config or {}
        self.device = torch.device(device)
        self.mae = TabularMAE(num_features=num_features, **mae_config)
        self.ocsvm = OCSVMDetector(**ocsvm_config)
        self.num_features = num_features
        self._mae_trained = False

    def freeze_encoder(self) -> None:
        """Freeze MAE parameters for embedding extraction."""
        for p in self.mae.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.mae.parameters():
            p.requires_grad = True

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings using frozen MAE encoder."""
        self.mae.eval()
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            emb = self.mae.get_embeddings(t)
            return emb.cpu().numpy()

    def fit_ocsvm(self, X_benign: np.ndarray) -> "HybridDetector":
        """
        Train OCSVM on benign embeddings.
        Assumes MAE is already trained and frozen.
        """
        embeddings = self.get_embeddings(X_benign)
        self.ocsvm.fit(embeddings)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 0 (normal) / 1 (anomaly)."""
        embeddings = self.get_embeddings(X)
        return self.ocsvm.predict_binary(embeddings)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        embeddings = self.get_embeddings(X)
        return self.ocsvm.evaluate(embeddings, y_true)

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.mae.state_dict(), path / "mae.pt")
        self.ocsvm.save(str(path / "ocsvm.joblib"))

    def load(self, path: str) -> "HybridDetector":
        path = Path(path)
        self.mae.load_state_dict(torch.load(path / "mae.pt", map_location=self.device))
        self.ocsvm = OCSVMDetector.load(str(path / "ocsvm.joblib"))
        self._mae_trained = True
        self.freeze_encoder()
        return self
