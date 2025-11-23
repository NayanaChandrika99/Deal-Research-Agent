"""Tests for DealGraph Bench dataset loader."""

from dealgraph.eval.bench_dataset import BenchQuery, load_dealgraph_bench
from dealgraph.data.ingest import load_all


def test_load_default_bench_dataset():
    """Default loader should return typed BenchQuery objects with valid deals."""
    bench_queries = load_dealgraph_bench()
    assert bench_queries, "Expected bench dataset to contain at least one query"
    assert all(isinstance(query, BenchQuery) for query in bench_queries)
    
    dataset = load_all("data/raw")
    deal_ids = {deal.id for deal in dataset.deals}
    
    for query in bench_queries:
        assert query.text.strip()
        assert query.relevant_deal_ids, "Each bench query should have labels"
        assert set(query.relevant_deal_ids).issubset(deal_ids)
