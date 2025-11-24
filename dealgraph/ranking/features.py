# ABOUTME: Feature extraction helpers for converting CandidateDeals to vectors.
# ABOUTME: Provides consistent feature ordering for the ranking model.

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..data.schemas import CandidateDeal


FEATURE_NAMES: List[str] = [
    "text_similarity",
    "sector_match",
    "region_match",
    "num_addons",
    "has_exit",
    "degree",
    "is_platform",
    "sector_degree",
    "region_degree",
    "out_degree",
    "in_degree",
    "description_length",
    "has_investment_year",
    "has_exit_year",
    "deal_age_years",
    "is_mature_deal",
    "has_complete_data",
    "text_graph_alignment",
]


def candidate_to_features(candidate: CandidateDeal) -> np.ndarray:
    """Convert a CandidateDeal into a numeric feature vector."""
    features = candidate.graph_features or {}
    values = []
    for name in FEATURE_NAMES:
        if name == "text_similarity":
            values.append(float(candidate.text_similarity))
        else:
            values.append(float(features.get(name, 0.0)))
    return np.array(values, dtype=np.float32)


def build_feature_matrix(candidates: Iterable[CandidateDeal]) -> np.ndarray:
    """Stack feature vectors for multiple candidates."""
    rows = [candidate_to_features(candidate) for candidate in candidates]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack(rows)
