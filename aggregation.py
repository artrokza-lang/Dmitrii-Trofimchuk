"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).
"""

from __future__ import annotations

import torch

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Aggregate hidden states into a feature vector.

    Uses mean pooling over the assistant's response tokens (masked by attention_mask)
    and concatenates the pooled representations from the last 4 transformer layers.

    Args:
        hidden_states: (n_layers, seq_len, hidden_dim) = (25, seq_len, 896)
        attention_mask: 1-D tensor with 1 for assistant response tokens, 0 otherwise.
                        Originally on CPU; will be moved to the same device as hidden_states.

    Returns:
        Feature vector of shape (4 * hidden_dim,) = (3584,)
    """
    # Ensure attention_mask is on the same device as hidden_states
    device = hidden_states.device
    attention_mask = attention_mask.to(device)

    # Take the last 4 layers (indices -4, -3, -2, -1)
    layers = hidden_states[-4:]              # (4, seq_len, 896)

    pooled_features = []
    for layer in layers:
        # Zero out non-response tokens (prompt and system)
        masked = layer * attention_mask.unsqueeze(-1)   # (seq_len, 896)
        sum_emb = masked.sum(dim=0)                     # (896,)
        num_tokens = attention_mask.sum().clamp(min=1)  # number of response tokens
        mean_pooled = sum_emb / num_tokens
        pooled_features.append(mean_pooled)

    # Concatenate the 4 layer vectors -> (3584,)
    return torch.cat(pooled_features)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric features (not used in final solution)."""
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Main entry point called from solution.py."""
    agg = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg, geo], dim=0)
    return agg
