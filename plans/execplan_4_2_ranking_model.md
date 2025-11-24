# ExecPlan 4.2 – Ranking Model Training & Serving

This plan adheres to `.agent/PLANS.md`. It is the sole reference needed to deliver Phase 4.2.

## Purpose / Big Picture

Now that we can generate synthetic (query, candidate, label) records, we need a baseline ranking model that converts candidate features into relevance scores. After this plan, we will have:
- Feature extraction utilities (`dealgraph/ranking/features.py`) that map `CandidateDeal` to a numeric vector.
- A lightweight ranking model (`dealgraph/ranking/model.py`, e.g., GradientBoosting or XGBoost) with save/load helpers.
- A training script (`dealgraph/ranking/train.py`) that ingests the synthetic dataset and produces `models/deal_ranker_v1.pkl`.
- Unit tests validating feature shapes, model training, and serialization.

This sets the stage for Phase 4.3, where we integrate the ML ranker into the agent and compare against the heuristic baseline.

## Progress

- [x] (2025-02-14 21:05Z) Feature extraction module implemented/tests added.
- [x] (2025-02-14 21:20Z) Ranking model class (fit/predict/save/load) complete.
- [x] (2025-02-14 21:35Z) Training script writes `models/deal_ranker_v1.pkl` from synthetic data.

## Surprises & Discoveries

- Stratified train/validation splits require at least two samples per class; for tiny synthetic slices (tests), the trainer now falls back to using the full dataset and logs the metric as a proxy.

## Decision Log

- Decision: Use scikit-learn `GradientBoostingRegressor` for the initial ranker and serialize via joblib to `models/deal_ranker_v1.pkl`.  
  Rationale: lightweight, no GPU dependency, stable API, good for small tabular datasets.  
  Date/Author: 2025-02-14 / Claude

## Outcomes & Retrospective

- Feature extraction, model training, and CLI automation are all in place with targeted pytest coverage. Phase 4.3 can now focus on integrating `DealRanker` into the agent and comparing against the heuristic baseline.

## Context and Orientation

- Input data: `data/processed/ranking_training_data.json` produced by Phase 4.1.
- Candidate deals can be obtained via existing retrieval path (`dealgraph.agent.tools.tool_graph_semantic_search`) or by reconstructing features directly; tests will mock `CandidateDeal`.
- Feature list is defined in SPEC §5.4.1; we already compute some graph/Text features in `dealgraph/retrieval/features.py`. We should reuse or import those helpers to avoid duplication.
- Model persistence path: `models/deal_ranker_v1.pkl` (per SPEC).
- CLI/training script is run via `python -m dealgraph.ranking.train`.

## Plan of Work

1. **Feature extraction (`dealgraph/ranking/features.py`)**
   - Define `FEATURE_NAMES` array (matching SPEC order: text_similarity, sector_match, region_match, num_addons, has_exit, degree…).
   - Implement `candidate_to_features(candidate: CandidateDeal) -> np.ndarray` using existing graph features (fallback to 0 if missing).
   - Provide helper `build_feature_matrix(candidates: List[CandidateDeal]) -> np.ndarray`.
   - Unit tests ensure ordering, dtype, handling of missing feature keys.

2. **Model class (`dealgraph/ranking/model.py`)**
   - Use scikit-learn `GradientBoostingRegressor` (or similar) for a first-pass ranker.
   - Implement `DealRanker.fit(X, y)`, `predict_scores(X)`, and persistence (`save(path)`, `load(path)`).
   - Add method `rank(candidates: List[CandidateDeal]) -> List[RankedDeal]`.
   - Tests: fit on toy data, check monotonic score ordering, serialization round-trip.

3. **Training pipeline (`dealgraph/ranking/train.py`)**
   - Load synthetic dataset, reconstruct features:
     - For each record, fetch `CandidateDeal` data (requires light dataset on disk or rehydration stub); for Phase 4.2, we can approximate by building pseudo-candidates directly from training data metadata or by joining with `load_all`.
     - Split into train/validation (e.g., 80/20) with deterministic seed.
   - Train `DealRanker`, evaluate simple metrics (accuracy, AUC, or mean average precision on validation split) and log them.
   - Save model to `models/deal_ranker_v1.pkl` (create directory if needed).
   - CLI options: `--data`, `--output`, `--max-records`, `--random-state`.

4. **Tests & Verification**
   - `tests/test_ranking_features.py`: feature array shape, deterministic output.
   - `tests/test_ranking_model.py`: training + save/load + scoring order.
   - `tests/test_ranking_train.py`: run training on a miniature dataset (monkeypatch load functions) and assert model file is written.
   - After unit tests pass, run CLI end-to-end in dry mode (e.g., `python -m dealgraph.ranking.train --max-records 50 --output models/deal_ranker_v1.pkl`).

## Concrete Steps

1. Create `dealgraph/ranking/features.py` implementing feature helpers; write tests.
2. Implement `dealgraph/ranking/model.py` with `DealRanker`.
3. Build `dealgraph/ranking/train.py` CLI + training logic.
4. Add test modules under `tests/` (three files as noted above); run `PYTHONPATH=. .venv/bin/pytest tests/test_ranking_features.py tests/test_ranking_model.py tests/test_ranking_train.py`.
5. Update documentation (`PHASES.md` Phase 4.2 section) once complete.

## Validation and Acceptance

- Targeted pytest suites pass.
- Manual training run produces `models/deal_ranker_v1.pkl` plus a log of validation metrics.
- `DealRanker.rank()` returns candidates sorted by predicted score and sets `rank` fields.

## Idempotence and Recovery

- Training CLI overwrites model file unless `--no-overwrite` is specified.
- Random seeds ensure reproducible splits; log seed/parameters in stdout and optionally metadata file under `models/`.
- Tests rely on mocked datasets to avoid heavy data generation; failures leave no partial artifacts besides temp files.

## Artifacts and Notes

- `models/deal_ranker_v1.pkl` (scikit-learn pickle).
- Optional `models/deal_ranker_v1.metadata.json` describing training config.
- Logs summarizing validation metric(s).

## Interfaces and Dependencies

- `FEATURE_NAMES: List[str]`
- `candidate_to_features(candidate: CandidateDeal) -> np.ndarray`
- `class DealRanker: fit(X, y); predict_scores(X); rank(candidates) -> List[RankedDeal]; save(path); load(path)`
- `python -m dealgraph.ranking.train --data data/processed/ranking_training_data.json --output models/deal_ranker_v1.pkl`
