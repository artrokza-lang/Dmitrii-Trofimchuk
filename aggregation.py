"""
aggregation.py — Variant E: last token final layer + norm of last token per layer
"""

from __future__ import annotations
import torch

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    attention_mask = attention_mask.to(device)
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    # Last token of final layer
    return hidden_states[-1][last_pos]            # (hidden_dim,)

def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    attention_mask = attention_mask.to(device)
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    norms = []
    for layer in hidden_states:                  # iterate over all layers
        vec = layer[last_pos]                    # (hidden_dim,)
        norms.append(torch.norm(vec))
    return torch.stack(norms)                    # (n_layers,)

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = True,                  # включено по умолчанию
) -> torch.Tensor:
    agg = aggregate(hidden_states, attention_mask)
    geo = extract_geometric_features(hidden_states, attention_mask).to(agg.device)
    return torch.cat([agg, geo])
