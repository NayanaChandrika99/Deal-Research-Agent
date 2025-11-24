"""Tests for ranking feature extraction."""

import numpy as np

from dealgraph.ranking.features import FEATURE_NAMES, candidate_to_features, build_feature_matrix
from dealgraph.data.schemas import Deal, CandidateDeal


def _make_candidate(**graph_overrides):
    deal = Deal(
         id="d1",
         name="Deal 1",
         sector_id="software",
         region_id="us",
         is_platform=True,
         status="current",
         description="Test deal",
     )
    features = {
        name: idx
        for idx, name in enumerate(FEATURE_NAMES[1:], start=1)
    }
    features.update(graph_overrides)
    return CandidateDeal(
        deal=deal,
        snippets=[],
        text_similarity=0.5,
        graph_features=features,
    )


def test_candidate_to_features_order():
    candidate = _make_candidate()
    vec = candidate_to_features(candidate)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert vec[0] == 0.5
    assert vec[1] == 1  # sector_match default from enumerate


def test_build_feature_matrix_handles_empty():
    matrix = build_feature_matrix([])
    assert matrix.shape == (0, len(FEATURE_NAMES))


def test_build_feature_matrix_stacks_rows():
    candidates = [_make_candidate(), _make_candidate(text_graph_alignment=9)]
    matrix = build_feature_matrix(candidates)
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert np.allclose(matrix[0, 0], 0.5)
    assert matrix[1, -1] == 9
