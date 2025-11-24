"""Integration and tool-level tests for agent orchestration helpers."""

from unittest.mock import patch

import pytest

from dealgraph.agent import tools as agent_tools
from dealgraph.agent.orchestrator import run_agent, AgentOrchestratorError
from dealgraph.data.schemas import (
    Deal,
    CandidateDeal,
    RankedDeal,
    Precedent,
    DealReasoningOutput,
)


class TestAgentToolsReasoning:
    """Validate ranking and reasoning tool wrappers."""

    def test_tool_rank_deals_orders_results(self):
        """Ranking tool should convert candidate dicts into RankedDeal objects."""
        candidates = [
            {
                "deal_id": "deal_low",
                "deal_name": "Low Score Deal",
                "sector_id": "industrial",
                "region_id": "eu",
                "is_platform": False,
                "status": "current",
                "description": "Industrial services",
                "text_similarity": 0.4,
                "graph_features": {"sector_match": 0},
                "snippets": [],
                "relevance_score": 0.25,
            },
            {
                "deal_id": "deal_high",
                "deal_name": "High Score Deal",
                "sector_id": "software",
                "region_id": "us",
                "is_platform": True,
                "status": "current",
                "description": "Software platform",
                "text_similarity": 0.92,
                "graph_features": {"sector_match": 1},
                "snippets": [],
                "relevance_score": 0.84,
            },
        ]

        ranked = agent_tools.tool_rank_deals("software query", candidates)

        assert isinstance(ranked[0], RankedDeal)
        assert ranked[0].candidate.deal.id == "deal_high"
        assert ranked[0].rank == 1
        assert ranked[1].candidate.deal.id == "deal_low"
        assert ranked[1].rank == 2

    def test_tool_deal_reasoner_invokes_reasoner(self):
        """Reasoner tool should delegate to deal_reasoner and return its output."""
        ranked = [
            RankedDeal(
                candidate=CandidateDeal(
                    deal=Deal(
                        id="deal_high",
                        name="High Score Deal",
                        sector_id="software",
                        region_id="us",
                        is_platform=True,
                        status="current",
                        description="Software platform",
                    ),
                    snippets=[],
                    text_similarity=0.9,
                    graph_features={"sector_match": 1},
                ),
                score=0.9,
                rank=1,
            )
        ]

        fake_output = DealReasoningOutput(
            precedents=[
                Precedent(
                    deal_id="deal_high",
                    name="High Score Deal",
                    similarity_reason="Matches query intent",
                )
            ],
            playbook_levers=["Roll-up strategy"],
            risk_themes=["Integration complexity"],
            narrative_summary="Compelling precedent.",
        )

        with patch("dealgraph.agent.tools.deal_reasoner", return_value=fake_output) as mock_reasoner:
            result = agent_tools.tool_deal_reasoner("software query", ranked)

        mock_reasoner.assert_called_once()
        assert result == fake_output

    def test_retrieval_to_reasoning_flow(self, monkeypatch):
        """Full flow: search -> rank -> reasoning should succeed with stubs."""
        # Force a clean index build using deterministic hashing encoder fallback.
        agent_tools._search_manager.reset()
        agent_tools.build_search_index(force_rebuild=True)

        # Patch deal_reasoner to avoid external LLM calls.
        fake_output = DealReasoningOutput(
            precedents=[
                Precedent(
                    deal_id="platform_tech_001",
                    name="Alpha Platform",
                    similarity_reason="Sector + strategy match",
                )
            ],
            playbook_levers=["Cybersecurity add-ons"],
            risk_themes=["Customer concentration"],
            narrative_summary="Alpha Platform is a strong precedent.",
        )

        with patch("dealgraph.agent.tools.deal_reasoner", return_value=fake_output):
            search_results = agent_tools.tool_graph_semantic_search(
                query="US software platform executing cybersecurity add-ons",
                top_k=3,
                include_explanations=False,
            )
            ranked = agent_tools.tool_rank_deals("Software query", search_results)
            reasoning = agent_tools.tool_deal_reasoner("Software query", ranked)

        assert ranked, "Expected ranked deals to be produced"
        assert isinstance(reasoning, DealReasoningOutput)
        assert reasoning.precedents[0].deal_id == "platform_tech_001"


class TestAgentOrchestrator:
    def test_run_agent_success(self, monkeypatch):
        fake_candidates = [
            {
                "deal_id": "deal_1",
                "deal_name": "Deal 1",
                "sector_id": "software",
                "region_id": "us",
                "is_platform": True,
                "status": "current",
                "description": "desc",
                "text_similarity": 0.9,
                "graph_features": {"sector_match": 1},
                "snippets": [],
                "relevance_score": 0.8,
            }
        ]
        monkeypatch.setattr(agent_tools, "tool_graph_semantic_search", lambda **kwargs: fake_candidates)
        monkeypatch.setattr(agent_tools, "tool_rank_deals_ml", lambda query, candidates, **kwargs: agent_tools.tool_rank_deals(query, candidates))

        fake_reasoning = DealReasoningOutput(
            precedents=[Precedent(deal_id="deal_1", name="Deal 1", similarity_reason="match")],
            playbook_levers=["lever"],
            risk_themes=["risk"],
            narrative_summary="Summary",
        )
        monkeypatch.setattr(agent_tools, "tool_deal_reasoner", lambda *args, **kwargs: fake_reasoning)

        log = run_agent("query", max_results=1)
        assert log.reasoning_output.precedents[0].deal_id == "deal_1"
        assert [call.tool for call in log.tool_calls] == [
            "tool_graph_semantic_search",
            "tool_rank_deals_ml",
            "tool_deal_reasoner",
        ]

    def test_run_agent_failure_when_no_candidates(self, monkeypatch):
        monkeypatch.setattr(agent_tools, "tool_graph_semantic_search", lambda **kwargs: [])
        with pytest.raises(AgentOrchestratorError):
            run_agent("query")
