"""Retrieval layer for DealGraph - graph-aware semantic search."""

from .features import (
    compute_graph_features,
    compute_batch_graph_features,
    extract_query_context,
    enhance_features_with_text,
    get_feature_names,
    validate_features
)
from .graph_search import (
    graph_semantic_search,
    rank_candidates,
    batch_search,
    search_with_explanations,
    get_search_explanation,
    build_snippets_index,
    validate_search_parameters
)

__all__ = [
    # Feature computation
    "compute_graph_features",
    "compute_batch_graph_features", 
    "extract_query_context",
    "enhance_features_with_text",
    "get_feature_names",
    "validate_features",
    # Search functionality
    "graph_semantic_search",
    "rank_candidates",
    "batch_search",
    "search_with_explanations",
    "get_search_explanation",
    "build_snippets_index",
    "validate_search_parameters"
]
