"""
probe.py — MLP classifier with threshold tuning
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


class HallucinationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self._net = None
        self._scaler = StandardScaler()
        self._threshold = 0.5

    def _build_network(self, input_dim: int):
        self._net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Call fit() first")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)
        self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # Positive weight to balance classes
        pos_weight = torch.tensor([(len(y) - y.sum()) / (y.sum() + 1e-8)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)

        # Train for fixed epochs (no early stopping for simplicity)
        for epoch in range(200):
            self.train()
            optimizer.zero_grad()
            loss = criterion(self(X_t), y_t)
            loss.backward()
            optimizer.step()

        self.eval()
        # Tune threshold on training set (or you can keep 0.5)
        with torch.no_grad():
            probs = torch.sigmoid(self(X_t)).cpu().numpy()
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.3, 0.71, 0.02):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(y, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        self._threshold = best_thresh
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        X_val_scaled = self._scaler.transform(X_val)
        X_val_t = torch.from_numpy(X_val_scaled).float()
        with torch.no_grad():
            probs = torch.sigmoid(self(X_val_t)).cpu().numpy()
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.3, 0.71, 0.02):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        self._threshold = best_thresh

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        self.eval()
        with torch.no_grad():
            logits = self(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1 - probs, probs]).T
