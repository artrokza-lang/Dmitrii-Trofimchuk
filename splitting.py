"""
splitting.py — Train / validation / test split utilities (student-implementable).
"""

import numpy as np
from sklearn.model_selection import train_test_split

def split_data(
    y: np.ndarray,
    df=None,                     # kept for compatibility, not used
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform a single stratified split into train, validation, and test sets.

    Returns:
        List containing one tuple (train_indices, val_indices, test_indices)
    """
    idx = np.arange(len(y))

    # First separate test set
    idx_train_val, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )

    # Then split the remaining into train and validation
    relative_val = val_size / (1.0 - test_size)   # proportion within train+val
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=y[idx_train_val]
    )

    return [(idx_train, idx_val, idx_test)]
