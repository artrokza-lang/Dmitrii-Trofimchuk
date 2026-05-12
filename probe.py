"""
probe.py — HallucinationProbe binary classifier (student-implemented).
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
        self._val_X = None
        self._val_y = None

    def _build_network(self, input_dim: int):
        """MLP with two hidden layers, dropout for regularisation."""
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
        """Train the probe with early stopping using validation set if provided."""
        X_scaled = self._scaler.fit_transform(X)
        self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # Positive weight to balance classes (neg/pos ratio)
        pos_weight = torch.tensor([(len(y) - y.sum()) / (y.sum() + 1e-8)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)

        # Prepare validation tensors if validation data was stored
        if self._val_X is not None:
            X_val_scaled = self._scaler.transform(self._val_X)
            X_val_t = torch.from_numpy(X_val_scaled).float()
            y_val_t = torch.from_numpy(self._val_y.astype(np.float32))

        best_loss = float('inf')
        patience = 5
        no_improve = 0
        best_state = None

        for epoch in range(300):
            self.train()
            optimizer.zero_grad()
            logits = self(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

            if self._val_X is not None:
                self.eval()
                with torch.no_grad():
                    val_logits = self(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve = 0
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if best_state is not None:
                            self.load_state_dict(best_state)
                        break
                self.train()

        self.eval()
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Store validation data and tune the decision threshold to maximise F1."""
        self._val_X = X_val.copy()
        self._val_y = y_val.copy()

        # After fit, we already have the model; now find best threshold
        X_val_scaled = self._scaler.transform(X_val)
        X_val_t = torch.from_numpy(X_val_scaled).float()
        with torch.no_grad():
            logits = self(X_val_t)
            probs = torch.sigmoid(logits).cpu().numpy()

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
        """Return binary labels using the tuned threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates (class 0, class 1)."""
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        self.eval()
        with torch.no_grad():
            logits = self(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1 - probs, probs]).T
