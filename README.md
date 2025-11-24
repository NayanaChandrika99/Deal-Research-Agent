# DealGraph Agent

A **Graph-Augmented RAG Agent** for private equity deal research.

## Documentation
*   [**SPECIFICATION.md**](./SPECIFICATION.md): The Product Requirements Document (PRD).
*   [**ARCHITECTURE.md**](./ARCHITECTURE.md): Technical design and component diagrams.
*   [**TEST_PLAN.md**](./tests/TEST_PLAN.md): Testing strategy.

## Quick Start

### Installation
```bash
pip install -e .
```

### Usage
```bash
# Run the full agent (retrieval -> ranking -> reasoning)
dealgraph-agent "US industrial distribution roll-up with add-on program" \
  --max-results 10 \
  --prompt-version latest \
  --output results/sample_run.json
```

The CLI prints the structured reasoning JSON followed by the narrative summary. When `--output` is supplied, the entire `AgentLog` (retrieval/ranking metadata) is persisted for later analysis.

### Ranking Workflow
Generate synthetic training data, train the ranker, and compare ML vs heuristic ranking:

```bash
# 1. Generate reverse-query labeled examples
python -m dealgraph.ranking.data_gen \
  --clusters 50 \
  --queries-per-cluster 3 \
  --negatives-per-query 2 \
  --output data/processed/ranking_training_data.json \
  --dry-run   # remove this flag to invoke the LLM

# 2. Train DealRanker
python -m dealgraph.ranking.train \
  --data data/processed/ranking_training_data.json \
  --output models/deal_ranker_v1.pkl

# 3. Benchmark ML vs baseline ranking
python -m dealgraph.eval.compare_ranking \
  --model models/deal_ranker_v1.pkl \
  --output results/ranking_comparison.json
```

### Tests

Run the full suite (or the subsets relevant to your changes):

```bash
PYTHONPATH=. .venv/bin/pytest
```

## Development
See `.agent/PLANS.md` for active development tasks.
