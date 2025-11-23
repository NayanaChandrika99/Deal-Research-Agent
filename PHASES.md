# Phased Implementation Plan

This document outlines the roadmap for building the DealGraph Agent.
We follow a **Data-First** approach: Graph → Search → Reasoning → Ranking.

Each Phase is broken down into atomic **Execution Plans (ExecPlans)** that will be written to `.agent/PLANS.md` when ready to start.

## Quick Reference

Each ExecPlan includes:
- **References**: Specific sections in SPECIFICATION.md, ARCHITECTURE.md, or other docs
- **Deliverables**: Concrete files and functionality to implement
- **Status**: [ ] Not Started | [✅] Complete

**Current Status**: Phase 2.1 Complete (Embeddings & Config)

## Progress Overview

| Phase | ExecPlans | Status |
|-------|-----------|--------|
| Phase 1: Foundation | 1.1, 1.2, 1.3 | Not Started |
| Phase 2: Retrieval | 2.1, 2.2, 2.3 | 2.1 ✅ Complete |
| Phase 3: Reasoning | 3.1, 3.2, 3.3, 3.4 | Not Started |
| Phase 4: Ranking | 4.1, 4.2, 4.3 | Not Started |
| Phase 5: Polish | 5.1, 5.2, 5.3 | Not Started |

---

## Phase 1: Foundation (Data & Graph)
**Goal**: Define the world model and load data into memory.

### [ ] ExecPlan 1.1: Data Schemas

**References**:
- SPECIFICATION.md § 4 ("Data Model & Schemas")
- SPECIFICATION.md § 4.1 ("Core entities")
- ARCHITECTURE.md § 3 ("Data Model")

**Deliverables**:
- `dealgraph/data/schemas.py` with Pydantic models:
  - `Deal`, `Sector`, `Region`, `Event`, `Snippet`
  - `CandidateDeal`, `RankedDeal`
  - `DealReasoningOutput`, `Precedent`
- Unit tests validating schema constraints

---

### [ ] ExecPlan 1.2: Data Ingestion

**References**:
- SPECIFICATION.md § 5.1.1 (`dealgraph/data/ingest.py`)
- SPECIFICATION.md § 3 (Repository Layout - `data/raw/`)

**Deliverables**:
- `dealgraph/data/ingest.py` with functions:
  - `load_sectors()`, `load_regions()`, `load_deals()`
  - `load_events()`, `load_snippets()`, `load_all()`
- Basic validation (no missing IDs, referential integrity)
- Unit tests for loading logic
- Sample synthetic data in `data/raw/`

---

### [ ] ExecPlan 1.3: Graph Builder

**References**:
- SPECIFICATION.md § 5.1.2 (`dealgraph/data/graph_builder.py`)
- ARCHITECTURE.md § 3 ("Graph Schema")
- ARCHITECTURE.md § 7 ADR-001 ("NetworkX for Graph Storage")

**Deliverables**:
- `dealgraph/data/graph_builder.py` with `DealGraph` class:
  - `add_deals()`, `add_sectors()`, `add_regions()`, `add_events()`, `add_snippets()`
  - `get_deal_neighbors()` - returns neighbors by type
- NetworkX MultiDiGraph with edges:
  - `IN_SECTOR`, `IN_REGION`, `ADDON_TO`, `EXITED_VIA`, `DESCRIBED_IN`
- Integration test verifying graph connectivity

---

## Phase 2: The "Golden Path" (Retrieval)
**Goal**: Enable basic search functionality.

### [✅] ExecPlan 2.1: Embeddings & Vector Index

**References**:
- SPECIFICATION.md § 5.2.1 (`dealgraph/embeddings/encoder.py`)
- SPECIFICATION.md § 5.2.2 (`dealgraph/embeddings/index.py`)
- SETUP.md ("OpenAI Embeddings API")

**Status**: ✅ COMPLETE
- `dealgraph/embeddings/encoder.py` - OpenAI text-embedding-3-small wrapper
- `dealgraph/config/settings.py` - Configuration management
- `test_api_connection.py` - API validation script

**Next**: Implement `DealEmbeddingIndex` for vector storage and search

---

### [ ] ExecPlan 2.2: Graph Search Logic

**References**:
- SPECIFICATION.md § 5.3.1 (`dealgraph/retrieval/features.py`)
- SPECIFICATION.md § 5.3.2 (`dealgraph/retrieval/graph_search.py`)
- ARCHITECTURE.md § 2 (Component Diagram - "Retrieval Layer")

**Deliverables**:
- `dealgraph/retrieval/features.py`:
  - `compute_graph_features()` - extract sector_match, region_match, num_addons, has_exit, degree
- `dealgraph/retrieval/graph_search.py`:
  - `graph_semantic_search()` - combine embedding similarity + graph features
  - Returns `List[CandidateDeal]` with text_similarity and graph_features
- Unit tests for feature computation

---

### [ ] ExecPlan 2.3: End-to-End Retrieval

**References**:
- SPECIFICATION.md § 5.6.1 (`dealgraph/agent/tools.py`)
- SPECIFICATION.md § 8 (Testing)

**Deliverables**:
- `dealgraph/agent/tools.py`:
  - `tool_graph_semantic_search()` wrapper
- Integration test:
  - Query: "Software consolidation play"
  - Verify: Returns software deals with correct features
  - Assert: Top results have high sector_match scores

---

## Phase 3: The Brain (Reasoning & DSPy)
**Goal**: Generate narrative answers using LLMs.

### [ ] ExecPlan 3.1: Reasoning Module (Naive Baseline)

**References**:
- SPECIFICATION.md § 5.5.2 (`dealgraph/reasoning/prompts.py`)
- SPECIFICATION.md § 5.5.3 (`dealgraph/reasoning/llm_client.py`)
- SPECIFICATION.md § 5.5.6 (`dealgraph/reasoning/reasoner.py`)
- PROMPT_OPTIMIZATION.md § "Start Simple"

**Deliverables**:
- `dealgraph/reasoning/prompts.py`:
  - `PromptRegistry` class for loading versioned prompts
  - `DEAL_REASONER_NAIVE_PROMPT` baseline template
- `dealgraph/reasoning/llm_client.py`:
  - `LLMClient` wrapper for Cerebras API
  - DSPy LM configuration
- `dealgraph/reasoning/reasoner.py`:
  - `deal_reasoner()` function using naive prompt
  - Returns `DealReasoningOutput` with precedents, playbook_levers, risk_themes, narrative
- `prompts/deal_reasoner/v1_naive.txt` - hand-written baseline
- Unit tests with mocked LLM responses

---

### [ ] ExecPlan 3.2: DealGraph Bench & Metrics

**References**:
- SPECIFICATION.md § 6 ("Evaluation: DealGraph Bench")
- SPECIFICATION.md § 6.1 ("Dataset")
- SPECIFICATION.md § 6.2 ("Metrics")
- PROMPT_OPTIMIZATION.md § "Composite Metric Design"

**Deliverables**:
- `dealgraph/eval/bench_dataset.py`:
  - `BenchQuery` schema
  - `load_dealgraph_bench()` function
- `dealgraph/eval/metrics.py`:
  - `recall_at_k()`, `ndcg_at_k()`, `precision_at_k()`
- `data/bench/bench_queries.json`:
  - 20-30 labeled queries with relevant_deal_ids
- Unit tests for metric calculations

---

### [ ] ExecPlan 3.3: DSPy Optimization Pipeline

**References**:
- SPECIFICATION.md § 5.5.4 (`dealgraph/reasoning/dspy_modules.py`)
- SPECIFICATION.md § 5.5.5 (`dealgraph/reasoning/optimizer.py`)
- SPECIFICATION.md § 10 ("Prompt Optimization Workflow")
- PROMPT_OPTIMIZATION.md (entire document)
- ARCHITECTURE.md § 6 ("Prompt Management Architecture")
- ARCHITECTURE.md § 7 ADR-002 ("DSPy MIPRO for Prompt Optimization")

**Deliverables**:
- `dealgraph/reasoning/dspy_modules.py`:
  - `DealReasonerSignature` - DSPy signature with input/output fields
  - `DealReasonerModule` - DSPy module with ChainOfThought
- `dealgraph/reasoning/optimizer.py`:
  - `DealReasonerMetric` class - composite metric (40% precision + 30% playbook + 30% narrative)
  - `optimize_deal_reasoner()` - MIPRO optimization function
  - LLM-as-judge implementations for quality scoring
- CLI script: `python -m dealgraph.reasoning.optimizer`
- `prompts/deal_reasoner/v2_optimized.json` - DSPy-optimized prompt
- `prompts/deal_reasoner/CHANGELOG.md` - version history with metrics

---

### [ ] ExecPlan 3.4: Reasoning Integration & Evaluation

**References**:
- SPECIFICATION.md § 5.5.6 (`reasoner.py` - updated with DSPy loading)
- SPECIFICATION.md § 6.3 ("Comparison Script")

**Deliverables**:
- Update `dealgraph/reasoning/reasoner.py`:
  - Load optimized DSPy module (v2) if available
  - Fallback to naive prompt (v1)
  - Fail loudly on errors (`DealReasonerError`)
- `dealgraph/eval/compare_prompts.py`:
  - Compare v1 (naive) vs v2 (optimized) on DealGraph Bench
  - Output: Precision@3, Playbook Quality, Narrative Coherence, Composite Score
- Integration test:
  - Query → Retrieval → Reasoning → Structured Output
  - Verify JSON format and narrative quality

---

## Phase 4: The Muscle (ML Ranking)
**Goal**: Improve precision using trained rankers.

### [ ] ExecPlan 4.1: Reverse-Query Data Generator

**References**:
- SPECIFICATION.md § 5.4.3 (`dealgraph/ranking/data_gen.py`)
- ARCHITECTURE.md § 4 ("Ranking Strategy" - Training Data Strategy Option B)

**Deliverables**:
- `dealgraph/ranking/data_gen.py`:
  - `generate_synthetic_training_data()` - sample deal clusters, use LLM to generate queries
  - Returns `List[dict]` with query, candidate_id, label (0/1)
- Generate ~1000 training examples
- Store in `data/processed/ranking_training_data.json`

---

### [ ] ExecPlan 4.2: Ranking Model

**References**:
- SPECIFICATION.md § 5.4.1 (`dealgraph/ranking/features.py`)
- SPECIFICATION.md § 5.4.2 (`dealgraph/ranking/model.py`)
- SPECIFICATION.md § 5.4.4 (`dealgraph/ranking/train.py`)

**Deliverables**:
- `dealgraph/ranking/features.py`:
  - `FEATURE_NAMES` list
  - `candidate_to_features()` - convert CandidateDeal to numpy array
- `dealgraph/ranking/model.py`:
  - `DealRanker` class with `fit()`, `predict_scores()`, `save()`, `load()`
  - Use XGBoost or scikit-learn GradientBoosting
- `dealgraph/ranking/train.py`:
  - `train_ranker_from_bench()` - train on DealGraph Bench + synthetic data
- Trained model saved to `models/deal_ranker_v1.pkl`
- Training script: `python -m dealgraph.ranking.train`

---

### [ ] ExecPlan 4.3: Ranking Integration & Evaluation

**References**:
- SPECIFICATION.md § 5.4.5 ("Ranking Tool")
- SPECIFICATION.md § 5.6.1 (`dealgraph/agent/tools.py` - updated)
- SPECIFICATION.md § 6.3 (`dealgraph/eval/compare_ranking.py`)

**Deliverables**:
- `dealgraph/ranking/__init__.py`:
  - `rank_deals()` function - takes candidates, returns ranked deals
- Update `dealgraph/agent/tools.py`:
  - `tool_rank_deals()` wrapper
- `dealgraph/eval/compare_ranking.py`:
  - Compare baseline (embedding-only) vs ML ranker
  - Metrics: Recall@k, NDCG@k for k=[1,3,5,10]
  - Output comparison table
- Integration test verifying ranking improves over baseline

---

## Phase 5: Polish (CLI & Agent)
**Goal**: User experience and orchestration.

### [ ] ExecPlan 5.1: Agent Orchestrator

**References**:
- SPECIFICATION.md § 5.6.2 (`dealgraph/agent/orchestrator.py`)
- ARCHITECTURE.md § 2 (Component Diagram - full pipeline)

**Deliverables**:
- `dealgraph/agent/orchestrator.py`:
  - `AgentLog` schema
  - `run_agent()` function orchestrating:
    1. `tool_graph_semantic_search()`
    2. `tool_rank_deals()`
    3. `tool_deal_reasoner()`
  - Logging for each step
- End-to-end integration test with real query

---

### [ ] ExecPlan 5.2: CLI Interface

**References**:
- SPECIFICATION.md § 7 ("CLI")
- SPECIFICATION.md § 7 (`dealgraph/cli/main.py`)
- README.md (Usage example)

**Deliverables**:
- `dealgraph/cli/main.py`:
  - Typer app with `query` command
  - Rich formatting for output
  - Options: `--prompt-version`, `--max-deals`, `--output-json`
- Console script entry point: `dealgraph-agent`
- Usage examples in README.md
- Help text and error messages

---

### [ ] ExecPlan 5.3: Documentation & Testing

**References**:
- SPECIFICATION.md § 8 ("Testing")
- SPECIFICATION.md § 9 ("Implementation Priorities")

**Deliverables**:
- Complete test suite:
  - `tests/test_data_ingest.py`
  - `tests/test_graph_search.py`
  - `tests/test_ranking_model.py`
  - `tests/test_reasoner.py`
  - `tests/test_prompt_optimization.py`
  - `tests/test_agent_orchestrator.py`
- All tests passing with >80% coverage
- Update README.md with:
  - Installation instructions
  - Quick start guide
  - Example queries
  - Architecture overview
- Update ARCHITECTURE.md with any implementation learnings
