"""Tests for prompt comparison tooling."""

import pytest

from dealgraph.data.schemas import (
    Deal,
    CandidateDeal,
    RankedDeal,
    Precedent,
    DealReasoningOutput,
)
from dealgraph.eval.bench_dataset import BenchQuery
from dealgraph.eval import compare_prompts


def _make_ranked_deal(deal_id: str) -> RankedDeal:
    deal = Deal(
        id=deal_id,
        name=f"Deal {deal_id}",
        sector_id="software",
        region_id="us",
        is_platform=True,
        status="current",
        description="Synthetic deal for testing.",
    )
    candidate = CandidateDeal(
        deal=deal,
        snippets=[],
        text_similarity=0.9,
        graph_features={"sector_match": 1},
    )
    return RankedDeal(candidate=candidate, score=0.9, rank=1)


class TestPromptComparator:
    """Unit tests for PromptComparator aggregate logic."""

    def test_comparator_aggregates_metrics(self):
        """Running comparator should average metrics across bench queries."""
        bench_queries = [
            BenchQuery(id="q1", text="Query 1", relevant_deal_ids=["deal_a"]),
            BenchQuery(id="q2", text="Query 2", relevant_deal_ids=["deal_b"]),
        ]

        def fake_search(query: BenchQuery):
            return [_make_ranked_deal(query.relevant_deal_ids[0])]

        def fake_reasoner(query_text: str, ranked_deals, prompt_version: str):
            summary = f"{prompt_version}:{query_text}"
            return DealReasoningOutput(
                precedents=[
                    Precedent(
                        deal_id=ranked_deals[0].candidate.deal.id,
                        name=ranked_deals[0].candidate.deal.name,
                        similarity_reason="matches",
                    )
                ],
                playbook_levers=["lever"],
                risk_themes=["risk"],
                narrative_summary=summary,
            )

        def fake_metrics(reasoning_output: DealReasoningOutput, bench_query: BenchQuery):
            version = reasoning_output.narrative_summary.split(":")[0]
            base_score = 0.4 if version == "v1" else 0.7
            return {
                "precision_at_3": base_score,
                "playbook_quality": base_score + 0.1,
                "narrative_coherence": base_score + 0.2,
            }

        comparator = compare_prompts.PromptComparator(
            bench_queries=bench_queries,
            search_runner=fake_search,
            reasoner_runner=fake_reasoner,
            metrics_evaluator=fake_metrics,
        )

        result = comparator.run("v1", "v2")

        assert result["queries_evaluated"] == 2
        assert result["baseline"]["precision_at_3"] == pytest.approx(0.4)
        assert result["candidate"]["precision_at_3"] == pytest.approx(0.7)
        assert result["delta"]["precision_at_3"] == pytest.approx(0.3)

    def test_compare_versions_wrapper_uses_comparator(self, monkeypatch):
        """compare_versions should delegate to default comparator factory."""
        called = {}

        class FakeComparator:
            def run(self, baseline_version, candidate_version, max_queries=None):
                called["baseline"] = baseline_version
                called["candidate"] = candidate_version
                called["max_queries"] = max_queries
                return {"queries_evaluated": 0}

        def fake_factory(max_queries=None):
            called["factory_max"] = max_queries
            return FakeComparator()

        monkeypatch.setattr(compare_prompts, "create_default_comparator", fake_factory)

        result = compare_prompts.compare_versions("v1", "v2", max_queries=5)

        assert result["queries_evaluated"] == 0
        assert called["baseline"] == "v1"
        assert called["candidate"] == "v2"
        assert called["factory_max"] == 5
