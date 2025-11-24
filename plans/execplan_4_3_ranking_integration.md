++ plans/execplan_4_3_ranking_integration.md
# ExecPlan 4.3 – ML Ranking Integration & Evaluation

This plan is governed by `.agent/PLANS.md`. It is self-contained: follow it from top to bottom to integrate the trained ranker and evaluate it against the baseline.

## Purpose / Big Picture

We now have a synthetic training dataset (Phase 4.1) and a trained `DealRanker` (Phase 4.2). Phase 4.3 wires the ranker into the agent pipeline, exposes tooling to compare ML vs heuristic rankings on DealGraph Bench, and adds regression tests ensuring ranking improves or at least behaves deterministically. After this plan, calling the agent will use `DealRanker` by default (with fallbacks) and we can quantify ranking gains using standard retrieval metrics.

## Progress

- [x] (2025-02-14 22:00Z) Ranker wiring in `dealgraph.agent.tools` + integration tests.
- [x] (2025-02-14 22:15Z) Benchmark comparison script and metrics CLI.
- [x] (2025-02-14 22:20Z) Tests covering integration & CLI evaluation.

## Surprises & Discoveries

- ML ranker loading needs caching with reset hooks for tests; added `reset_cached_ranker()` to avoid cross-test contamination.

## Decision Log

- Decision: Keep heuristic ranking as fallback when ML ranker fails/missing; ensures agent remains usable even if model file isn’t distributed.  
  Rationale: Prevents runtime crashes in air-gapped environments and simplifies CI.  
  Date/Author: 2025-02-14 / Claude

## Outcomes & Retrospective

- Agent now has ML ranking tooling with deterministic tests, and `compare_ranking.py` provides a benchmark-compatible CLI. Next steps will involve Phase 5 orchestration where ranking and reasoning are chained end-to-end.

## Context and Orientation

- Ranker artifacts: `models/deal_ranker_v1.pkl`.
- Candidate data: results from `tool_graph_semantic_search` (already includes graph features).
- Retrieval metrics: `dealgraph.eval.metrics` (precision@k, recall@k, ndcg@k).
- Bench dataset: `data/bench/bench_queries.json`.
- We already have `dealgraph.eval.compare_prompts`; we’ll add a sibling `compare_ranking.py`.

## Plan of Work

1. **Agent integration**
   - Extend `dealgraph.agent.tools` with:
     - `load_ranker()` helper to lazy-load `models/deal_ranker_v1.pkl`.
     - `tool_rank_deals_ml(query, candidates, model_path=None)` returning `RankedDeal`s (default path from settings).
     - Ensure fallbacks: if model load fails, revert to the existing heuristic `_compute_relevance_score`.
   - Update orchestrator or future pipeline entry points to call the ML ranker by default (if orchestrator not yet present, add a stub or update documentation for Phase 5).

2. **Benchmark comparison script**
   - Create `dealgraph/eval/compare_ranking.py` with CLI: `python -m dealgraph.eval.compare_ranking --model models/deal_ranker_v1.pkl --data data/bench/bench_queries.json`.
   - Workflow:
     - For each Bench query, run retrieval to get candidates.
     - Produce baseline ordering (current heuristic ranking from `tool_graph_semantic_search`).
     - Apply ML ranker (convert dicts to `CandidateDeal` objects, run `tool_rank_deals_ml`).
     - Compute metrics via `precision_at_k`, `recall_at_k`, `ndcg_at_k` for k in {1,3,5,10}.
     - Output aggregate metrics and per-query deltas to stdout or optional JSON.

3. **Tests**
   - `tests/test_rank_integration.py` (new):
     - Mock ranker + model file to ensure `tool_rank_deals_ml` sorts correctly and handles fallback.
     - Test benchmark script function with stubbed retrieval and metrics to avoid expensive operations.
   - End-to-end smoke test: use a tiny dataset to ensure CLI writes metrics file (mock retrieval to skip API calls).

4. **Docs & Phase updates**
   - Update `PHASES.md` and this ExecPlan’s Progress/Outcomes once done.
   - Optionally document ranking usage in README (brief mention).

## Concrete Steps

1. Implement ranker loader/utilities in `dealgraph.agent.tools`.
2. Create `dealgraph/eval/compare_ranking.py` + CLI.
3. Write tests:
   - Unit tests for `tool_rank_deals_ml` (mocking model load).
   - Tests for comparator logic (mock retrieval + fake scores).
4. Run pytest: `PYTHONPATH=. .venv/bin/pytest tests/test_rank_integration.py`.
5. Manual CLI smoke run (dry-run retrieval using stub data if necessary).

## Validation and Acceptance

- Tests pass (targeted suite + regression).
- CLI outputs metrics JSON for baseline vs ML ranker, showing consistent behavior.
- Agent tools can return `RankedDeal`s using ML scores by default.

## Idempotence and Recovery

- Ranker loader caches model; have a `reset_ranker()` for tests.
- Benchmark CLI takes `--output` and over-writes by default.
- Retrieval falls back to heuristic ranking if model file missing or scoring fails.

## Artifacts and Notes

- `dealgraph/eval/compare_ranking.py` (CLI).
- `tests/test_rank_integration.py`.
- Model loading log entries for observability.

## Interfaces and Dependencies

- `tool_rank_deals_ml(query: str, candidates: List[Dict[str, Any]], model_path: Optional[str] = None) -> List[RankedDeal]`
- `python -m dealgraph.eval.compare_ranking --model models/deal_ranker_v1.pkl --output results/ranking_comparison.json`
