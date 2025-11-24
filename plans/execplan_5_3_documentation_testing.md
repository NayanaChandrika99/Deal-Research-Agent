++ plans/execplan_5_3_documentation_testing.md
# ExecPlan 5.3 – Documentation & Testing

This plan follows `.agent/PLANS.md` and covers final hardening: ensuring the test matrix is in place, docs are updated, and the system is ready for handoff.

## Purpose / Big Picture

With all functional phases complete, this phase ensures:
1. The test suite covers every major component (data, graph, retrieval, ranking, reasoning, orchestrator, CLI).
2. README/SETUP instructions reflect the current capabilities (CLI usage, training workflow).
3. Optional: produce a lightweight TEST_PLAN or summary (if not already tracked).

After this plan, a contributor can `pip install -e .`, run `pytest`, and use the CLI following README.

## Progress

- [x] (2025-11-23 20:25Z) Reviewed PHASES.md, this ExecPlan, README, and ran targeted pytest modules to confirm graph search and benchmark regressions.
- [x] (2025-11-23 20:38Z) Added a shared deal-graph mock helper plus explanation fallback, hardened performance test generation/statistics, and filtered stray third-party warnings.
- [x] (2025-11-23 20:46Z) Executed the full `PYTHONPATH=. .venv/bin/pytest` suite and refreshed PHASES.md + CHANGES_SUMMARY.md to capture the finished Phase 5 state.

## Surprises & Discoveries

- Observation: `Mock(spec=DealGraph)` does not expose a `.graph` attribute, so every graph-search test that configured `deal_graph.graph.has_node` failed before asserting behavior.  
  Evidence: `tests/test_graph_search.py` raised `AttributeError: Mock object has no attribute 'graph'` during initial pytest run.
- Observation: `TestCaseGenerator.generate_test_suite()` returned zero cases whenever `num_cases < len(categories)`, which cascaded into `ZeroDivisionError` inside `BenchmarkSuite` and broke the integration tests.  
  Evidence: `tests/test_performance_benchmarks.py::TestBenchmarkSuite::test_run_comprehensive_benchmark_mock` hit `success_count / len(test_cases)` with `len(test_cases) == 0`.
- Observation: `scipy.stats.ttest_rel` emits a precision-loss warning (treated as test noise) when baseline and optimized series have identical differences.  
  Evidence: `tests/test_performance_benchmarks.py::TestStatisticalAnalysis::test_paired_ttest` produced the RuntimeWarning until we short-circuited on near-zero variance.

## Decision Log

- Decision: Create a reusable `build_mock_deal_graph()` helper inside `tests/test_graph_search.py` instead of stubbing `.graph` ad hoc or loosening the production `DealGraph` interface.  
  Rationale: Keeps the production code unchanged while reducing test duplication and ensuring every test configures the NetworkX degrees consistently.  
  Date/Author: 2025-11-23 / Claude

- Decision: Over-generate at least one benchmark test case per category (using `math.ceil`) and trim after shuffling rather than honoring the exact `num_cases // categories` split that frequently produced an empty suite.  
  Rationale: Guarantees statistically valid inputs for BenchmarkSuite even during tiny CI runs while preserving the requested total via slicing.  
  Date/Author: 2025-11-23 / Claude

- Decision: Filter the external `pydantic.warnings.PydanticDeprecatedSince20` warning via `pyproject.toml` instead of vendoring or patching litellm.  
  Rationale: Keeps pytest output pristine without modifying upstream dependencies; future dependency bumps can remove the filter.  
  Date/Author: 2025-11-23 / Claude

## Outcomes & Retrospective

The plan delivered a green, warning-free pytest run (236 tests) plus documentation that reflects reality: README already covered the CLI workflow, while PHASES.md and CHANGES_SUMMARY.md now mark every ExecPlan through Phase 5 as complete. The new graph-search helper eliminated brittle mocks, the benchmark generator/statistics guardrails stabilized tiny test suites, and the pytest filter keeps third-party deprecation noise out of CI logs. With these changes the agent can be installed, trained, and queried end-to-end following only the README.

## Context and Orientation

- The repository already contains exhaustive suites for ingestion, retrieval, ranking, reasoning, orchestrator, CLI, prompt optimization, and performance benchmarking; a few of them regressed (graph search mocks, benchmark generator, statistical analysis warnings) after the Phase 5 additions.
- README is current: it documents installation, CLI usage, ranking data generation, and the `dealgraph-agent` workflow, so no substantive README edits were required.
- Phase artifacts (PHASES.md, CHANGES_SUMMARY.md) still referenced partially complete phases and needed to be updated once the tests passed.
- Goal: stabilize the outstanding tests, ensure pytest output is warning-free, then refresh the roadmap/docs so a newcomer can trust both the CLI instructions and the status tables.

## Plan of Work

1. **Stabilize failing tests**
   - Add a reusable helper inside `tests/test_graph_search.py` so mocks expose `.graph` and update `get_search_explanation()` to backfill `text_similarity` from the candidate when the features dict omits it.
   - Ensure `dealgraph/performance/test_cases.py` always emits at least one case per category (and marks its classes `__test__ = False`) so BenchmarkSuite never divides by zero; update `dealgraph/performance/metrics.py` to short-circuit the paired t-test when the variance of differences is ~0 to avoid SciPy warnings.
   - Filter the upstream Pydantic deprecation warning via `[tool.pytest.ini_options].filterwarnings` instead of patching dependencies.

2. **Full-suite verification**
   - Run targeted modules to prove the fixes fail-before/pass-after, then execute `PYTHONPATH=. .venv/bin/pytest` to capture an artifact showing all 236 tests passing without warnings.

3. **Documentation refresh**
   - README already contains CLI/ranking instructions; focus on synchronizing PHASES.md and CHANGES_SUMMARY.md with the newly completed ExecPlans (5.1–5.3) and test guarantee.
   - Record the completed work in this ExecPlan (progress, discoveries, decision log, retrospective).

## Concrete Steps

1. Reproduce the regressions from the repository root:  
   `PYTHONPATH=. .venv/bin/pytest tests/test_graph_search.py tests/test_performance_benchmarks.py`
2. Implement the graph-search helper, explanation fallback, benchmark/test-case fixes, statistical guardrails, and pytest warning filter.
3. Run the full suite for the acceptance artifact:  
   `PYTHONPATH=. .venv/bin/pytest`
4. Update PHASES.md, CHANGES_SUMMARY.md, and this ExecPlan to reflect the completed state of Phase 5.

## Validation and Acceptance

- `PYTHONPATH=. .venv/bin/pytest` must report `236 passed` without warnings (after the targeted modules reproduce the pre-fix failures).
- README already demonstrates CLI/ranking usage; validation is that PHASES.md and CHANGES_SUMMARY.md now mark ExecPlans 5.1–5.3 as complete with accurate descriptions.
- This ExecPlan reflects the progress/decision log so a cold-start contributor can see what changed and why.

## Idempotence and Recovery

- README edits are additive.
- Tests are rerunnable.

## Interfaces and Dependencies

- CLI: `dealgraph-agent "US industrial roll-up" --output results/run.json`.
- Data pipeline: `python -m dealgraph.ranking.data_gen ...`, `python -m dealgraph.ranking.train ...`.
