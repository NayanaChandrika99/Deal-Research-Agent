# Test Plan

This document outlines the testing strategy for the DealGraph Agent.
All new features must be accompanied by a test case in the appropriate section.

## 1. Unit Tests

### Data Ingestion (`tests/test_data_ingest.py`)
*   [ ] **Load Sectors**: Verify `load_sectors` parses valid JSON/CSV.
*   [ ] **Load Deals**: Verify `load_deals` correctly links `sector_id`.
*   [ ] **Schema Validation**: Ensure missing required fields raise validation errors (Pydantic).

### Graph Builder (`tests/test_graph_builder.py`)
*   [ ] **Node Creation**: Verify all entities exist as nodes.
*   [ ] **Edge Creation**: Verify `IN_SECTOR`, `ADDON_TO` edges are created correctly.
*   [ ] **Neighborhood**: Verify `get_deal_neighbors` returns correct adjacent nodes.

### Retrieval Features (`tests/test_retrieval_features.py`)
*   [ ] **Sector Match**: If query mentions "Software", deal in "Software" gets `sector_match=1`.
*   [ ] **Add-on Count**: Verify `num_addons` matches graph degree for `ADDON_TO` edges.
*   [ ] **Exit Flag**: Verify `has_exit` is true if `EXITED_VIA` edge exists.

### Ranking Model (`tests/test_ranking_model.py`)
*   [ ] **Feature Shape**: Verify feature vector has correct dimensions.
*   [ ] **Scoring Logic**: Verify that a deal with higher similarity + better graph features gets a higher score.
*   [ ] **Training Loop**: Verify `DealRanker.fit` runs without error on dummy data.

## 2. Integration Tests

### Agent Flow (`tests/test_agent_flow.py`)
*   [ ] **Mocked Pipeline**: Run `run_agent` with:
    *   Mock Retrieval (returns fixed 5 candidates).
    *   Mock Ranker (reorders them).
    *   Mock Reasoner (returns dummy JSON).
    *   **Goal**: Ensure data flows correctly between components.

## 3. Evaluation / Benchmark

### Offline Bench (`tests/test_benchmark.py`)
*   [ ] **Recall Check**: Run against `data/bench/bench_queries.json` and assert Recall@10 > X% (sanity check).

