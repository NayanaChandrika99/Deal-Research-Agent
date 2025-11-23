# ABOUTME: Evaluation utilities for DealGraph benchmarks.
# ABOUTME: Exposes bench dataset helpers and metric calculations.

from .bench_dataset import BenchQuery, load_dealgraph_bench
from .metrics import precision_at_k, recall_at_k, ndcg_at_k

__all__ = [
    "BenchQuery",
    "load_dealgraph_bench",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
]
