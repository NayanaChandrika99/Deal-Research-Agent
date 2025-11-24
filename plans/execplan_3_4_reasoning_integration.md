# ExecPlan 3.4 – Reasoning Integration & Evaluation

This ExecPlan follows the guidelines in `.agent/PLANS.md`. Treat it as the complete set of instructions required for a new contributor to finish Phase 3.4 without any other context.

## Purpose / Big Picture

Reasoning is currently decoupled from the rest of the agent: the optimized DSPy prompt is never executed, there is no way to compare prompt versions on DealGraph Bench, and no integration test exercises retrieval → reasoning. After this plan, we will have a production-ready reasoning module that automatically loads the DSPy program, falls back to the naive prompt when needed, exposes comparison tooling, and is wired into our agent/tooling with regression tests.

## Progress

- [x] (2025-02-14 18:05Z) Agent tool wrappers + end-to-end retrieval→reasoning integration test added.
- [x] (2025-02-14 18:30Z) Prompt comparison comparator + CLI scaffolded with unit tests.
- [x] (2025-02-14 18:45Z) Reasoner error handling aligned with spec (DealReasonerError alias) and plan ready for future DSPy wiring.

## Surprises & Discoveries

- Importing `dealgraph.reasoning.dspy.evaluator` implicitly loads `optimizer.py`; a bad relative import there (previously unused) broke comparator tests until corrected to jump three directories upward.

## Decision Log

- Decision: Exposed simple `tool_rank_deals`/`tool_deal_reasoner` in `dealgraph.agent.tools` instead of waiting for Phase 4 ranker so that integration tests can execute now.  
  Rationale: Needed deterministic RankedDeal objects immediately; heuristic ranking using existing relevance scores satisfies Phase 3.4 requirements.  
  Date/Author: 2025-02-14 / Claude

## Outcomes & Retrospective

- Reasoner toolchain now surfaces structured results through agent helpers, and the new comparator CLI provides a reproducible way to evaluate prompt versions on DealGraph Bench. Remaining DSPy work (actual module loading) will ride under ExecPlan 3.3 updates.

## Context and Orientation

Relevant paths:
- `dealgraph/reasoning/reasoner.py` currently implements only the naive prompt; it must be extended to load the optimized DSPy module (`DealReasonerModule`) plus a fallback path.
- DSPy utilities live under `dealgraph/reasoning/dspy/`. There is no top-level `dealgraph/reasoning/optimizer.py` or CLI.
- Metrics + bench dataset exist (`dealgraph/eval/metrics.py`, `dealgraph/eval/bench_dataset.py`).
- Retrieval tools (`dealgraph/agent/tools.py`) expose search helpers only; no ranking or reasoning entry points.
- Tests to touch/add: `tests/test_reasoning.py`, new `tests/test_compare_prompts.py`, `tests/test_agent_orchestrator.py` (create if needed) for integration.

## Plan of Work

1. **Reasoner Integration**
   - Update `dealgraph/reasoning/reasoner.py`:
     - Introduce `DealReasonerError` per SPEC §5.5.6.
     - Load DSPy optimized program from `prompts/deal_reasoner/v2_optimized.json` via `DealReasonerModule.load()` when `prompt_version="latest"` or an explicit JSON version exists.
     - Fallback chain: optimized → requested JSON version → naive prompt.
     - Add structured logging and validation using `DealReasoningOutput`.
   - Extend `tests/test_reasoning.py` to cover:
     - DSPy path (use a stub module or monkeypatch `DealReasonerModule`).
     - Explicit error raising when neither prompt path is available.

2. **Prompt Comparison CLI**
   - Add `dealgraph/eval/compare_prompts.py`:
     - Load DealGraph Bench via `load_dealgraph_bench`.
     - For each query, run retrieval (via `dealgraph.agent.tools.tool_graph_semantic_search` or lower-level helpers) to produce candidate deals, then call `deal_reasoner` with specified prompt versions.
     - Compute metrics (Precision@3, Playbook quality proxy, Narrative coherence proxy, Composite) using existing helpers or, if absent, extend `dealgraph/reasoning/dspy/evaluator.PerformanceEvaluator`.
     - Support CLI usage: `python -m dealgraph.eval.compare_prompts --baseline v1 --candidate v2 --output results/prompt_comparison.json`.
   - Unit test (`tests/test_compare_prompts.py`) should stub LLM/encoder calls to keep it deterministic and assert metrics aggregation and output serialization.

3. **Agent Tooling & Integration Test**
   - Extend `dealgraph/agent/tools.py` with:
     - `tool_deal_reasoner(query, ranked_deals, max_deals=10, prompt_version="latest")` that wraps `deal_reasoner` and returns a dict suitable for orchestrator use.
     - (Optional) `tool_rank_deals` stub returning ranked results (depending on ranking availability; for now, convert graph-search candidates into `RankedDeal` sorted by relevance).
   - Add an integration test covering retrieval → ranking placeholder → reasoning:
     - Under `tests/test_agent_orchestrator.py` or a new file, mock embedding/LLM layers to keep the pipeline deterministic.
     - Assert the reasoning output JSON structure plus logging metadata.

## Concrete Steps

1. Edit `dealgraph/reasoning/reasoner.py` per above; run `pytest tests/test_reasoning.py`.
2. Create `dealgraph/eval/compare_prompts.py` with Typer/argparse CLI; add tests under `tests/test_compare_prompts.py`; run targeted pytest module.
3. Update `dealgraph/agent/tools.py` and write integration test (`tests/test_agent_orchestrator.py` or similar); run `pytest tests/test_agent_orchestrator.py`.
4. After all targeted tests pass, run `pytest -k "reasoning or compare_prompts or agent"` to ensure coverage.

## Validation and Acceptance

- `pytest tests/test_reasoning.py` passes with new DSPy path tests.
- `pytest tests/test_compare_prompts.py` (new) passes and produces deterministic summaries.
- `pytest tests/test_agent_orchestrator.py` (new or updated) passes, proving end-to-end reasoning flow.
- Manual CLI dry run: `python -m dealgraph.eval.compare_prompts --baseline v1 --candidate v2 --max-queries 2 --output /tmp/out.json` produces metrics JSON referencing both prompt versions.

## Idempotence and Recovery

- All CLI scripts accept `--output` so repeated runs simply overwrite artifacts.
- Reasoner changes are backward-compatible because they fall back to the naive prompt when optimized assets are missing.
- Tests rely on mocks/fakes; rerunning them does not mutate repository state.

## Artifacts and Notes

- `results/prompt_comparison.json` (optional output) captures metric deltas; document its schema in the CLI module docstring.
- Consider logging prompt versions used in `dealgraph/reasoning/reasoner.py` for future monitoring hooks.

## Interfaces and Dependencies

- `dealgraph/reasoning/reasoner.deal_reasoner(query: str, ranked_deals: List[RankedDeal], max_deals: int = 10, prompt_version: str = "latest") -> DealReasoningOutput`
- `dealgraph/eval/compare_prompts.main(baseline: str, candidate: str, max_queries: int, output: Optional[str])`
- `dealgraph/agent/tools.tool_deal_reasoner(query: str, ranked_deals: List[RankedDeal], prompt_version: str = "latest") -> Dict[str, Any]`

Keep this plan updated as work proceeds: fill in Progress checkboxes with timestamps, note discoveries, and record decisions in the log.
