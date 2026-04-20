"""
Hybrid Detector: MAE (frozen) + One-Class SVM
Combines learned representations with explicit anomaly boundaries.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..torch_io import load_state_dict_checkpoint
from .mae import TabularMAE
from .ocsvm import OCSVMDetector, parse_ocsvm_section


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
        svm_kw, self._embedding_batch_size, self._max_fit_samples = parse_ocsvm_section(
            ocsvm_config
        )
        self.device = torch.device(device)
        self.mae = TabularMAE(num_features=num_features, **mae_config)
        self.ocsvm = OCSVMDetector(**svm_kw)
        self.num_features = num_features
        self._mae_trained = False

    def freeze_encoder(self) -> None:
        """Freeze MAE parameters for embedding extraction."""
        for p in self.mae.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.mae.parameters():
            p.requires_grad = True

    def get_embeddings(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Extract embeddings using frozen MAE encoder (batched to limit RAM)."""
        bs = batch_size or self._embedding_batch_size
        self.mae.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                batch = X[i : i + bs].astype(np.float32, copy=False)
                t = torch.from_numpy(batch).to(self.device)
                emb = self.mae.get_embeddings(t)
                outs.append(emb.cpu().numpy())
        return np.vstack(outs)

    def fit_ocsvm(self, X_benign: np.ndarray, seed: int = 42) -> "HybridDetector":
        """
        Train OCSVM on benign embeddings.
        Assumes MAE is already trained and frozen.
        """
        X_fit = X_benign
        cap = self._max_fit_samples
        if cap is not None and len(X_fit) > cap:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X_fit), size=cap, replace=False)
            X_fit = X_fit[idx]
            print(
                f"Hybrid: fitting OCSVM on {len(X_fit)} benign rows "
                f"(subsampled from {len(X_benign)}; set ocsvm.max_fit_samples to change)"
            )
        embeddings = self.get_embeddings(X_fit)
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
        self.mae.load_state_dict(load_state_dict_checkpoint(path / "mae.pt", self.device))
        self.ocsvm = OCSVMDetector.load(str(path / "ocsvm.joblib"))
        self._mae_trained = True
        self.freeze_encoder()
        return self
