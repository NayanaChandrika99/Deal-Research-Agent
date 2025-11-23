# ABOUTME: Evaluation metrics for retrieval and reasoning performance.
# ABOUTME: Provides precision, recall, and NDCG helpers.

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def precision_at_k(predicted: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Compute Precision@K."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for deal_id in predicted[:k] if deal_id in relevant_set)
    return hits / float(k)


def recall_at_k(predicted: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Compute Recall@K."""
    relevant_list = list(relevant)
    if not relevant_list or k <= 0:
        return 0.0
    relevant_set = set(relevant_list)
    hits = sum(1 for deal_id in predicted[:k] if deal_id in relevant_set)
    return hits / float(len(relevant_set))


def ndcg_at_k(predicted: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K."""
    if k <= 0:
        return 0.0
    relevant_list = list(relevant)
    if not relevant_list:
        return 0.0

    gains = _dcg(predicted[:k], set(relevant_list))
    ideal_gains = _dcg(relevant_list[:k], set(relevant_list))
    if ideal_gains == 0:
        return 0.0
    return gains / ideal_gains


def _dcg(predicted: Sequence[str], relevant_set: set[str]) -> float:
    score = 0.0
    for idx, deal_id in enumerate(predicted, start=1):
        if deal_id in relevant_set:
            score += 1.0 / math.log2(idx + 1)
    return score
