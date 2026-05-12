"""
aggregation.py — Variant D: mean pooling over all real tokens (last layer only)
"""

from __future__ import annotations
import torch

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    attention_mask = attention_mask.to(device)
    # Last transformer layer
    layer = hidden_states[-1]                     # (seq_len, hidden_dim)
    masked = layer * attention_mask.unsqueeze(-1) # zero out padding
    sum_emb = masked.sum(dim=0)
    num_tokens = attention_mask.sum().clamp(min=1)
    return sum_emb / num_tokens                   # (hidden_dim,)

def extract_geometric_features(...):
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
