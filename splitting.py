"""
splitting.py — 5‑fold stratified cross‑validation split
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

def split_data(
    y: np.ndarray,
    df=None,
    test_size: float = 0.15,   # not used directly, kept for compatibility
    val_size: float = 0.15,    # not used directly
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns a list of 5 train/val/test splits (k‑fold without a separate test set).
    For each fold, we use 4 folds for training and 1 fold for validation.
    The final test set (unlabelled) will be predicted using the model trained on the full dataset,
    so the split here only affects the reported validation metrics.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = []
    for train_index, val_index in skf.split(np.zeros(len(y)), y):
        # In this scheme, there is no separate test split; we treat val as the evaluation split.
        # We'll set test_index = val_index (so metrics are reported on val).
        splits.append((train_index, val_index, val_index))
    return splits
