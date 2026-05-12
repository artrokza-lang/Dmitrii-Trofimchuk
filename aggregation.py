"""
aggregation.py — Last token pooling from last 4 layers, concatenated
"""

from __future__ import annotations
import torch

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    hidden_states: (n_layers, seq_len, hidden_dim)
    attention_mask: (seq_len,) 1 for real tokens, 0 padding
    Returns: concatenated last-token features from last 4 layers (4 * hidden_dim,)
    """
    device = hidden_states.device
    attention_mask = attention_mask.to(device)

    # Find last real token index
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    # Take last 4 transformer layers (indices -4, -3, -2, -1)
    layers = hidden_states[-4:]   # each has shape (seq_len, hidden_dim)
    features = []
    for layer in layers:
        features.append(layer[last_pos])   # (hidden_dim,)
    return torch.cat(features)             # (4 * hidden_dim,)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Optional geometric features (not used in this version)."""
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg, geo.to(agg.device)])
    return agg
