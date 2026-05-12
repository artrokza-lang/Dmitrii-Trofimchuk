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
    """
    hidden_states: (n_layers, seq_len, hidden_dim)
    attention_mask: (seq_len,) — original padding mask from tokenizer (1 for real tokens)
    Returns: feature vector (4 * hidden_dim,)
    """
    device = hidden_states.device
    attention_mask = attention_mask.to(device)

    # Найдём длину реальной последовательности (без паддинга)
    seq_len = attention_mask.sum().long().item()

    # Возьмём последние 50% токенов (примерно половина — это ответ)
    # Минимум 1 токен
    response_len = max(1, seq_len // 2)
    start_idx = seq_len - response_len

    # Создадим маску: 1 для последних response_len токенов
    response_mask = torch.zeros_like(attention_mask)
    response_mask[start_idx:seq_len] = 1.0

    # Используем последние 4 слоя
    layers = hidden_states[-4:]  # (4, seq_len, 896)

    pooled_features = []
    for layer in layers:
        masked = layer * response_mask.unsqueeze(-1)
        sum_emb = masked.sum(dim=0)
        num_tokens = response_mask.sum().clamp(min=1)
        mean_pooled = sum_emb / num_tokens
        pooled_features.append(mean_pooled)

    return torch.cat(pooled_features)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Геометрические признаки: нормы слоёв, стандартное отклонение."""
    device = hidden_states.device
    attention_mask = attention_mask.to(device)

    # Нормы каждого слоя (среднее по токенам)
    norms = []
    for layer in hidden_states:
        # Учитываем только реальные токены
        masked = layer * attention_mask.unsqueeze(-1)
        sum_norm = masked.norm(dim=-1).sum()
        num_tokens = attention_mask.sum().clamp(min=1)
        norms.append((sum_norm / num_tokens).item())

    # Стандартное отклонение норм по слоям
    std_norm = torch.tensor(norms).std()

    return torch.tensor([std_norm])


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
