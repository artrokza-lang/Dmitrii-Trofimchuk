"""
aggregation.py — Last token pooling (no heuristic, uses final token of response)
"""

from __future__ import annotations
import torch

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    hidden_states: (n_layers, seq_len, hidden_dim)
    attention_mask: (seq_len,) 1 for real tokens, 0 for padding
    Returns: feature vector (hidden_dim,)
    """
    device = hidden_states.device
    attention_mask = attention_mask.to(device)

    # Find index of last real token (non-padding)
    real_positions = attention_mask.nonzero(as_tuple=False)  # (n_real, 1)
    last_pos = int(real_positions[-1].item())

    # Take final transformer layer and last token
    feature = hidden_states[-1][last_pos]  # (hidden_dim,)
    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Not used in this solution."""
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Main entry point called from solution.py."""
    agg = aggregate(hidden_states, attention_mask)
    return agg
