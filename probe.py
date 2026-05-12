"""
probe.py — HallucinationProbe binary classifier (improved version)
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
        self._net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
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

        pos_weight = torch.tensor([(len(y) - y.sum()) / (y.sum() + 1e-8)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        best_loss = float('inf')
        patience = 5
        no_improve = 0
        best_state = None

        for epoch in range(400):
            self.train()
            optimizer.zero_grad()
            loss = criterion(self(X_t), y_t)
            loss.backward()
            optimizer.step()

            if self._val_X is not None:
                X_val_scaled = self._scaler.transform(self._val_X)
                X_val_t = torch.from_numpy(X_val_scaled).float()
                y_val_t = torch.from_numpy(self._val_y.astype(np.float32))
                self.eval()
                with torch.no_grad():
                    val_loss = criterion(self(X_val_t), y_val_t).item()
                scheduler.step(val_loss)
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
        if self._val_X is None:
            with torch.no_grad():
                probs = torch.sigmoid(self(X_t)).cpu().numpy()
            best_f1, best_t = 0, 0.5
            for t in np.arange(0.3, 0.71, 0.02):
                f1 = f1_score(y, (probs >= t).astype(int))
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            self._threshold = best_t
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        self._val_X = X_val.copy()
        self._val_y = y_val.copy()
        X_val_scaled = self._scaler.transform(X_val)
        X_val_t = torch.from_numpy(X_val_scaled).float()
        with torch.no_grad():
            probs = torch.sigmoid(self(X_val_t)).cpu().numpy()
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.3, 0.71, 0.02):
            f1 = f1_score(y_val, (probs >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self._threshold = best_t

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
