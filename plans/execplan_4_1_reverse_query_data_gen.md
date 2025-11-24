# ExecPlan 4.1 – Reverse-Query Training Data Generator

This ExecPlan is governed by `.agent/PLANS.md`. Treat it as the complete specification for delivering Phase 4.1 with no other context.

## Purpose / Big Picture

Ranking requires labeled (query, candidate, label) tuples. We do not have human-labeled data, so we synthesize it by sampling clusters from the DealGraph and asking the LLM to invent realistic user queries (“reverse query” strategy). After this plan, we can deterministically generate ~1000 labeled examples stored under `data/processed/` that future ExecPlans (4.2/4.3) consume for feature engineering, model training, and evaluation.

## Progress

- [x] (2025-02-14 19:45Z) Data sampling utilities created (`DealCluster` extraction + sampler).
- [x] (2025-02-14 20:00Z) Query generation + labeling pipeline implemented with deterministic fallback.
- [x] (2025-02-14 20:05Z) CLI/tests verifying dataset schema and deterministic behavior.

## Surprises & Discoveries

- LLM failures need deterministic fallbacks so dataset generation works offline; implemented `_fallback_queries` to keep CLI usable in dry-run or when the API is unavailable.

## Decision Log

- Decision: Keep record schema lean (`query`, `candidate_id`, `label`, `cluster_id`, `metadata`) and persist metadata separately alongside the JSON file.  
  Rationale: downstream ranking code should read a simple list, while provenance (seed, counts) belongs in a sidecar metadata file to avoid bloating each record.  
  Date/Author: 2025-02-14 / Claude

## Outcomes & Retrospective

- Generated reproducible synthetic ranking datasets via `python -m dealgraph.ranking.data_gen`. Tests cover both generation and CLI dry-run paths; future ranking phases can now rely on `data/processed/ranking_training_data.json` without mocking.

## Context and Orientation

- Data sources: `data/raw/*.json` loaded via `dealgraph/data/ingest.py` and `DealGraph` from `dealgraph/data/graph_builder.py`.
- Ranking package currently empty; we will create `dealgraph/ranking/data_gen.py` + supporting utilities.
- LLM calls use `dealgraph/reasoning/llm_client.LLMClient` (already deterministic when mocked). For tests we must stub to keep them offline.
- Output location per SPEC §5.4.3: `data/processed/ranking_training_data.json` (JSON lines or list).
- Tests go under `tests/test_ranking_data_gen.py`.

## Plan of Work

1. **Graph sampler + cluster extraction**
   - Add helpers in `dealgraph/ranking/data_gen.py` (new file) to:
     - Load dataset via `load_all`.
     - Build DealGraph; extract platform + linked add-ons/events to form mini-clusters.
     - Convert clusters into lightweight dicts with names/sectors/regions/snippets (used in prompt).
   - Provide deterministic sampling (seeded `random.Random`) so tests are stable.

2. **Reverse query generator**
   - Implement `generate_synthetic_training_data(...)` signature returning `List[dict]` with `query`, `candidate_id`, `label`, optional metadata (cluster_id, reason, etc.).
   - Use `LLMClient` to request 3-5 realistic queries per cluster; parse JSON array string.
   - Treat cluster deals as positive labels (1) and sample negatives of same sector/regional diversity (label 0).
   - Provide knobs: `num_clusters`, `queries_per_cluster`, `negatives_per_query`, `seed`, `llm_client`.
   - Add caching/resume: optional `existing_data` path to append vs regenerate.

3. **Persistence + CLI**
   - Write helper `save_training_data(records, output_path)` storing canonical JSON list (indent=2 for readability).
   - Build CLI entry point `python -m dealgraph.ranking.data_gen --clusters 50 --output data/processed/ranking_training_data.json`.
   - CLI should skip LLM calls when `--dry-run` is passed (use stub queries).

4. **Testing**
   - `tests/test_ranking_data_gen.py`:
     - Mock `LLMClient.complete_json` to return deterministic query arrays.
     - Validate schema (each record has required fields, labels 0/1).
     - Check deterministic behavior given fixed seed (two runs produce identical results).
     - Assert CLI logic writes file and respects `--dry-run`.

5. **Documentation**
   - Update `PHASES.md` (Phase 4.1 deliverable section) after implementation.
   - Add short README section or docstring describing how to regenerate dataset (optional but recommended).

## Concrete Steps

1. Create `dealgraph/ranking/` package files if missing: `__init__.py`, `data_gen.py`.
2. Implement cluster sampler + generator functions with docstrings referencing SPEC §5.4.3.
3. Add CLI `main()` guarded by `if __name__ == "__main__":`.
4. Write tests with pytest + monkeypatch.
5. Run `PYTHONPATH=. .venv/bin/pytest tests/test_ranking_data_gen.py`.
6. Generate a small sample (e.g., `python -m dealgraph.ranking.data_gen --clusters 5 --dry-run --output /tmp/ranking_sample.json`) to verify CLI behavior manually.

## Validation and Acceptance

- `pytest tests/test_ranking_data_gen.py` passes locally and in CI.
- `python -m dealgraph.ranking.data_gen --clusters 10 --dry-run --output data/processed/ranking_training_data.json` creates a file with the expected schema (dry-run mode uses stub queries; real run may require API keys).
- Dataset contains at least 100 labeled examples with both positive and negative labels.
- Plan updated (Progress, Surprises, Decision Log, Outcomes) once each milestone finishes.

## Idempotence and Recovery

- CLI accepts `--overwrite` flag (default true) to avoid accidental data loss; if false and file exists, abort with a clear message.
- Generation seeds ensure reproducibility; log the seed in output metadata.
- Failures mid-generation should surface exceptions and leave partial files only when `--overwrite` is enabled; document this behavior.

## Artifacts and Notes

- Primary artifact: `data/processed/ranking_training_data.json`.
- Optional metadata file (same stem `.metadata.json`) describing generation parameters (seed, clusters, timestamp).
- Logging should include cluster IDs and query counts for observability.

## Interfaces and Dependencies

- `dealgraph.ranking.data_gen.generate_synthetic_training_data(num_clusters: int = 50, queries_per_cluster: int = 3, negatives_per_query: int = 2, seed: Optional[int] = None, llm_client: Optional[LLMClient] = None) -> List[dict]`
- CLI usage: `python -m dealgraph.ranking.data_gen --clusters 100 --queries-per-cluster 3 --negatives-per-query 3 --output data/processed/ranking_training_data.json`
- Output record schema: `{"query": str, "candidate_id": str, "label": int, "cluster_id": str, "metadata": {...}}`
