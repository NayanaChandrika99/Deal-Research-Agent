# ExecPlan 5.1 – Agent Orchestrator (Graph → Rank → Reason)

This plan follows `.agent/PLANS.md`. It is the canonical spec for delivering Phase 5.1.

## Purpose / Big Picture

We now have retrieval, ML ranking, and reasoning components. Phase 5.1 connects them into a single orchestrator that:
1. Accepts a user query.
2. Runs graph semantic search.
3. Applies ML ranking.
4. Invokes the reasoning module to produce structured/narrative answers.

After this plan, a single call to `run_agent(query)` will return the reasoning output plus logs of each tool call. This orchestration is also the foundation for the CLI (Phase 5.2) and full end-to-end tests (Phase 5.3).

## Progress

- [ ] (TBD) Orchestrator implemented (run_agent + AgentLog).
- [ ] (TBD) Tests validating tool call sequencing and error handling.
- [ ] (TBD) Logging metadata (inputs/outputs summaries) in place.

## Surprises & Discoveries

- None yet.

## Decision Log

- None yet.

## Outcomes & Retrospective

- To be filled upon completion.

## Context and Orientation

- Tooling available:
  - `tool_graph_semantic_search` / `tool_rank_deals_ml` / `tool_deal_reasoner`.
- We need a new module `dealgraph/agent/orchestrator.py` containing:
  - `AgentLog` schema (likely Pydantic BaseModel).
  - `run_agent(query, top_k=10, prompt_version="latest") -> AgentLog`.
  - Logging structure: list of tool calls, each with tool name, inputs summary, outputs summary (avoid large payloads).
- Error handling: orchestrator should raise a dedicated exception or include status in `AgentLog` if a tool fails.
- Tests belong in `tests/test_agent_orchestrator.py` (extend or new file) using mocks to avoid real API calls.

## Plan of Work

1. **Schema and orchestrator**
   - Create `dealgraph/agent/orchestrator.py` with:
     - `class AgentLog(BaseModel)` capturing query, tool_calls (list), reasoning_output.
     - `run_agent(query: str, *, max_results: int = 10, prompt_version: str = "latest")`.
     - Tool call order:
       1. `tool_graph_semantic_search`.
       2. `tool_rank_deals_ml` (on search results).
       3. `tool_deal_reasoner`.
     - Each step logs minimal input/output summaries (e.g., candidate count, top deal IDs).
     - Provide optional dependency injection for tests (e.g., `search_fn`, `rank_fn`, `reason_fn` parameters).

2. **Error handling**
   - Wrap each tool call; failures should raise an `AgentOrchestratorError` with context.
   - Log partial results even on failure (e.g., retrieval succeeded but ranking failed).

3. **Testing**
   - Expand `tests/test_agent_orchestrator.py` (existing integration tests) to cover:
     - run_agent success path (mocking retrieval/rank/reason functions).
     - run_agent failure path (e.g., ranking raises, ensure error surfaces).
     - Validate `AgentLog` contents (tool call order, summary fields).
   - Use deterministic mocks to avoid network calls.

4. **Validation**
   - Run targeted tests: `PYTHONPATH=. .venv/bin/pytest tests/test_agent_orchestrator.py`.
   - Manual dry run (optional): stub LM/encoder to ensure `run_agent("Sample query")` returns a log.

## Concrete Steps

1. Create `dealgraph/agent/orchestrator.py` with orchestrator logic/schema.
2. Update `dealgraph/agent/__init__.py` (if needed) to export `run_agent`.
3. Enhance tests to cover orchestrator behavior.
4. Update `PHASES.md` once complete.

## Validation and Acceptance

- Tests pass with new orchestrator coverage.
- `run_agent()` returns structured `AgentLog` with retrieval/ranking/reasoning data.
- Errors propagate with clear messages.

## Idempotence and Recovery

- Orchestrator uses existing tools that already cache search/ranker state; no additional caches introduced.
- Logging is ephemeral; no file I/O.
- Tests isolate state by mocking tool functions and using `reset_cached_ranker` when necessary.

## Interfaces and Dependencies

- `run_agent(query: str, max_results: int = 10, prompt_version: str = "latest") -> AgentLog`
- `AgentLog.tool_calls` entries: `{"tool": str, "inputs": str, "outputs": str}`
- Exceptions: `AgentOrchestratorError` raised on fatal failures.
