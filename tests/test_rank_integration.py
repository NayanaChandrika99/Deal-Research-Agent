"""Integration tests for ML ranking utilities."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from dealgraph.agent import tools as agent_tools
from dealgraph.data.schemas import Deal, CandidateDeal, RankedDeal
from dealgraph.ranking.model import DealRanker
from dealgraph.eval import compare_ranking


def test_tool_rank_deals_ml_falls_back(monkeypatch):
    candidates = [
        {
            "deal_id": "deal_1",
            "deal_name": "Deal One",
            "sector_id": "software",
            "region_id": "us",
            "is_platform": True,
            "status": "current",
            "description": "Desc",
            "text_similarity": 0.9,
            "graph_features": {"sector_match": 1},
            "snippets": [],
            "relevance_score": 0.5,
        }
    ]

    class BrokenRanker:
        def rank(self, _):
            raise RuntimeError("boom")

    monkeypatch.setattr(agent_tools, "_load_ranker", lambda path=None: BrokenRanker())
    ranked = agent_tools.tool_rank_deals_ml("query", candidates)
    assert ranked[0].candidate.deal.id == "deal_1"


def test_compare_ranking_cli(monkeypatch, tmp_path):
    bench_queries = [
        Mock(id="bench_query_1", text="software query", relevant_deal_ids=["deal_1"])
    ]
    monkeypatch.setattr(compare_ranking, "load_dealgraph_bench", lambda: bench_queries)

    # Mock search results
    monkeypatch.setattr(
        agent_tools,
        "tool_graph_semantic_search",
        lambda **kwargs: [
            {
                "deal_id": "deal_1",
                "deal_name": "Deal 1",
                "sector_id": "software",
                "region_id": "us",
                "is_platform": True,
                "status": "current",
                "description": "desc",
                "text_similarity": 0.8,
                "graph_features": {"sector_match": 1},
                "snippets": [],
                "relevance_score": 0.7,
            },
            {
                "deal_id": "deal_2",
                "deal_name": "Deal 2",
                "sector_id": "software",
                "region_id": "us",
                "is_platform": False,
                "status": "current",
                "description": "desc",
                "text_similarity": 0.4,
                "graph_features": {"sector_match": 0},
                "snippets": [],
                "relevance_score": 0.3,
            },
        ],
    )

    fake_ranker = Mock()
    fake_ranker.rank.return_value = [
        RankedDeal(
            candidate=CandidateDeal(
                deal=Deal(
                    id="deal_2",
                    name="Deal 2",
                    sector_id="software",
                    region_id="us",
                    is_platform=False,
                    status="current",
                    description="desc",
                ),
                snippets=[],
                text_similarity=0.4,
                graph_features={"sector_match": 0},
            ),
            score=0.4,
            rank=1,
        ),
        RankedDeal(
            candidate=CandidateDeal(
                deal=Deal(
                    id="deal_1",
                    name="Deal 1",
                    sector_id="software",
                    region_id="us",
                    is_platform=True,
                    status="current",
                    description="desc",
                ),
                snippets=[],
                text_similarity=0.8,
                graph_features={"sector_match": 1},
            ),
            score=0.8,
            rank=2,
        ),
    ]

    monkeypatch.setattr(agent_tools, "_load_ranker", lambda path=None: fake_ranker)
    output_path = tmp_path / "ranking_metrics.json"
    argv = [
        "compare_ranking",
        "--model",
        "models/deal_ranker_v1.pkl",
        "--max-queries",
        "1",
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    compare_ranking.main()
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data["queries_evaluated"] == 1
