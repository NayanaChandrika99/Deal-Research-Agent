"""Tests for DealRanker model."""

import numpy as np
from dealgraph.ranking.model import DealRanker
from dealgraph.data.schemas import Deal, CandidateDeal


def _candidate(score_factor: float) -> CandidateDeal:
    deal = Deal(
        id=f"deal_{score_factor}",
        name="Deal",
        sector_id="software",
        region_id="us",
        is_platform=True,
        status="current",
        description="Desc",
    )
    features = {
        "sector_match": score_factor,
        "region_match": 0,
        "num_addons": 0,
        "has_exit": 0,
        "degree": 0,
        "is_platform": 1,
        "sector_degree": 0,
        "region_degree": 0,
        "out_degree": 0,
        "in_degree": 0,
        "description_length": 10,
        "has_investment_year": 0,
        "has_exit_year": 0,
        "deal_age_years": 0,
        "is_mature_deal": 0,
        "has_complete_data": 1,
        "text_graph_alignment": score_factor,
    }
    return CandidateDeal(
        deal=deal,
        snippets=[],
        text_similarity=score_factor,
        graph_features=features,
    )


def test_ranker_fit_and_rank(tmp_path):
    X = np.array([[0.1], [0.5], [0.9]])
    X = np.hstack([X for _ in range(18)])  # match feature count
    y = np.array([0.1, 0.6, 0.9])
    ranker = DealRanker()
    ranker.fit(X, y)

    candidates = [_candidate(val) for val in (0.2, 0.8, 0.5)]
    ranked = ranker.rank(candidates)
    assert ranked[0].candidate.deal.id == "deal_0.8"

    model_path = tmp_path / "ranker.pkl"
    ranker.save(model_path)
    loaded = DealRanker.load(model_path)
    assert np.allclose(loaded.predict_scores(X), ranker.predict_scores(X))
