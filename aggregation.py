"""
aggregation.py — Variant F: last token from layers 12,16,20,24 (0‑based indices)
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
    # Qwen2.5‑0.5B has 24 layers, indices 0..23. Choose evenly spaced.
    selected = [11, 15, 19, 23]     # layers 12,16,20,24 (1‑based)
    features = [hidden_states[i][last_pos] for i in selected]
    return torch.cat(features)      # (4 * hidden_dim,)

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
