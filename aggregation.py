"""
aggregation.py — Last token pooling + geometric features
"""

from __future__ import annotations
import torch

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns last token's hidden state from the final layer.
    hidden_states: (n_layers, seq_len, hidden_dim)
    attention_mask: (seq_len,) with 1 for real tokens
    """
    device = hidden_states.device
    attention_mask = attention_mask.to(device)

    # Last real token index
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    # Take final transformer layer, last token
    feature = hidden_states[-1][last_pos]   # (hidden_dim,)
    return feature

def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Geometric features: mean norm per layer, std across layers, and difference between last two layers.
    Computed over real tokens only.
    """
    device = hidden_states.device
    attention_mask = attention_mask.to(device)
    n_layers, seq_len, hidden_dim = hidden_states.shape

    norms = []
    for layer in hidden_states:
        masked = layer * attention_mask.unsqueeze(-1)        # zero out padding
        token_norms = torch.norm(masked, dim=-1)             # (seq_len,)
        sum_norms = token_norms.sum()
        num_tokens = attention_mask.sum().clamp(min=1)
        mean_norm = sum_norms / num_tokens
        norms.append(mean_norm)

    norms = torch.stack(norms)                      # (n_layers,)
    std_norm = norms.std()                          # scalar
    diff_last_two = norms[-1] - norms[-2]           # scalar
    # Ratio of response length (heuristic – not needed, but can keep)
    # Instead, we use only reliable stats
    return torch.tensor([std_norm, diff_last_two])

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = True,
) -> torch.Tensor:
    agg = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo = extract_geometric_features(hidden_states, attention_mask)
        # Ensure geo is on same device as agg
        geo = geo.to(agg.device)
        return torch.cat([agg, geo])
    return agg
