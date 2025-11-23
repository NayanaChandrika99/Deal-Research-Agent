````markdown
# Project Spec — DealGraph Agentic Precedent Search

## 1. Project Overview

### 1.1 Goal

Build a **Deal Research Agent** for private-equity–style precedent and playbook queries.

Given a **free-text query** describing a deal or strategic question (e.g.,  
“US industrial distribution roll-up with multiple add-ons then strategic exit”),  
the system should:

1. **Retrieve** relevant historical deals and related text (news, case studies) using **graph-aware semantic search**.
2. **Rank** these deals with a **learned ML ranker** (not just cosine similarity).
3. **Call an LLM** (via prompting, no fine-tuning in v1) to:
   - Select the best precedents.
   - Extract common **playbook levers** and **risk themes**.
   - Return a structured + narrative answer.

This should be exposed as:

- A **Python library** (`dealgraph`) with clean modules and APIs.
- A **CLI** for running queries.
- A small **offline benchmark** to evaluate retrieval & ranking.

### 1.2 Non-goals (for v1)

- Training a custom LLM from scratch or LoRA fine-tuning (future extension).
- Building a full web UI.
- Connecting to real internal PE data (we use **public-like synthetic data** only).
- Implementing heavy distributed infrastructure (keep it single-node, local files or a lightweight DB).

---

## 2. High-Level Architecture

### 2.1 Components

1. **Data / Graph Layer**
   - Ingests a small corpus of:
     - Deals (portfolio companies)
     - Sectors
     - Regions
     - Events (add-ons, exits)
     - Text snippets (news, case study paragraphs)
   - Builds a **DealGraph**:
     - Nodes: `Deal`, `Sector`, `Region`, `Event`, `Snippet`
     - Edges reflecting relations: sector membership, geography, add-on relationships, exits, descriptions.
   - Stores:
     - Graph structure (e.g., NetworkX or simple adjacency).
     - Embeddings for text nodes (vector index).

2. **Graph-aware Retrieval Tool**
   - API: `graph_semantic_search(query: str) -> List[CandidateDeal]`
   - Uses:
     - Text embeddings for initial candidate retrieval.
     - Graph-derived features (sector match, num_addons, exit flags, etc.).

3. **Ranking Tool (ML ranker)**
   - API: `rank_deals(query: str, candidates: List[CandidateDeal]) -> List[RankedDeal]`
   - Uses a trained model to rank candidates based on:
     - Text similarity.
     - Graph features.
   - **Training Data Strategy**: Implements a "Reverse-Query" generator that inspects deal clusters (e.g., platform + add-ons) and uses an LLM to synthesize realistic user queries and relevance labels (Option B).
   - Compares against a **baseline** (embedding-only) using an offline benchmark.

4. **Reasoning Layer (Prompted LLM)**
   - API: `deal_reasoner(query: str, deals: List[RankedDeal]) -> DealReasoningOutput`
   - Calls a **frontier LLM via API** with prompt templates.
   - Produces:
     - Selected precedents.
     - Playbook levers.
     - Risk themes.
     - Narrative explanation.

5. **Agent Orchestrator**
   - API: `run_agent(query: str) -> DealResearchAnswer`
   - Orchestrates:
     - `graph_semantic_search` → `rank_deals` → `deal_reasoner`
   - Logs each step for analysis.

6. **Evaluation / Benchmark (DealGraph Bench)**
   - A small set of **queries with labeled precedents**.
   - Scripts to compute:
     - Recall@k and NDCG@k for:
       - Baseline (embedding-only) vs ML ranker.

---

## 3. Repository Layout

```text
dealgraph/
  __init__.py

  config/
    __init__.py
    settings.py          # paths, model names, constants

  data/
    __init__.py
    schemas.py           # dataclasses / pydantic models for Deal, Sector, Event, Snippet, CandidateDeal, etc.
    ingest.py            # load raw data (JSON/CSV/Markdown) -> normalized objects
    graph_builder.py     # build DealGraph from normalized data
    store.py             # simple persistence (e.g., pickle/parquet/SQLite)

  embeddings/
    __init__.py
    encoder.py           # wrapper around embedding provider (OpenAI, etc.)
    index.py             # vector index (e.g., FAISS or simple in-memory)

  retrieval/
    __init__.py
    graph_search.py      # graph_semantic_search() implementation
    features.py          # compute graph-aware features per candidate

  ranking/
    __init__.py
    features.py          # assemble feature vectors for ranker
    model.py             # ML ranker (train, save, load, predict)
    train.py             # training script using DealGraph Bench

  reasoning/
    __init__.py
    prompts.py           # naive baseline prompts + PromptRegistry
    llm_client.py        # wrapper around LLM API + DSPy configuration
    dspy_modules.py      # DSPy signatures and modules
    optimizer.py         # DSPy MIPRO optimization + metrics
    reasoner.py          # deal_reasoner() implementation

  agent/
    __init__.py
    tools.py             # thin wrappers so agent can call retrieval, ranking, reasoner
    orchestrator.py      # run_agent() function (tool-calling loop)

  eval/
    __init__.py
    bench_dataset.py     # definitions / loader for DealGraph Bench
    metrics.py           # Recall@k, NDCG@k, etc.
    compare_ranking.py   # script to compare baseline vs ML ranker

  cli/
    __init__.py
    main.py              # typer-based CLI, entrypoint `dealgraph-agent`

prompts/
  deal_reasoner/
    v1_naive.txt         # hand-written baseline
    v2_optimized.json    # DSPy MIPRO-optimized
    CHANGELOG.md         # semantic versioning log
  reverse_query/
    v1_naive.txt
    CHANGELOG.md

tests/
  test_data_ingest.py
  test_graph_search.py
  test_ranking_model.py
  test_reasoner.py
  test_prompt_optimization.py
  test_agent_orchestrator.py

PROJECT_SPEC.md
README.md
requirements.txt
pyproject.toml / setup.cfg
````

---

## 4. Data Model & Schemas

Use **Python dataclasses or Pydantic models**.

### 4.1 Core entities

```python
from typing import List, Optional
from pydantic import BaseModel

class Sector(BaseModel):
    id: str
    name: str  # e.g., "Healthcare", "Industrial Services"

class Region(BaseModel):
    id: str
    name: str  # e.g., "United States", "Europe"

class EventType(str):
    ADDON = "addon"
    EXIT = "exit"

class Event(BaseModel):
    id: str
    type: EventType
    deal_id: str               # deal this event belongs to
    related_deal_id: Optional[str] = None  # for ADDON_TO relationships
    date: Optional[str] = None            # ISO date as string
    description: Optional[str] = None

class Snippet(BaseModel):
    id: str
    deal_id: str
    source: str                # e.g., "news", "case_study"
    text: str

class Deal(BaseModel):
    id: str
    name: str
    sector_id: str
    region_id: str
    is_platform: bool
    status: str                # e.g., "current", "realized"
    description: str
    # optional fields:
    year_invested: Optional[int] = None
    year_exited: Optional[int] = None
```

### 4.2 Candidate & Ranked deal structs

```python
class CandidateDeal(BaseModel):
    deal: Deal
    snippets: List[Snippet]
    text_similarity: float      # cosine similarity
    graph_features: dict        # e.g., {"sector_match": 1, "num_addons": 3}

class RankedDeal(BaseModel):
    candidate: CandidateDeal
    score: float
    rank: int
```

### 4.3 Reasoning outputs

```python
class Precedent(BaseModel):
    deal_id: str
    name: str
    similarity_reason: str

class DealReasoningOutput(BaseModel):
    precedents: List[Precedent]
    playbook_levers: List[str]
    risk_themes: List[str]
    narrative_summary: str
```

---

## 5. Module-Level Specs

### 5.1 Data & Graph

#### 5.1.1 `dealgraph/data/ingest.py`

**Responsibilities:**

* Load raw deal, sector, region, events, snippets data from `data/raw/` (CSV, JSON, Markdown).
* Normalize into `Deal`, `Sector`, `Region`, `Event`, `Snippet` objects.
* Perform basic validation (no missing IDs, referential integrity).

**Functions:**

```python
def load_sectors(path: str) -> List[Sector]: ...
def load_regions(path: str) -> List[Region]: ...
def load_deals(path: str) -> List[Deal]: ...
def load_events(path: str) -> List[Event]: ...
def load_snippets(path: str) -> List[Snippet]: ...

def load_all(base_path: str) -> dict:
    """
    Returns dict with keys: sectors, regions, deals, events, snippets
    """
```

#### 5.1.2 `dealgraph/data/graph_builder.py`

**Responsibilities:**

* Build an in-memory graph from normalized data.
* Use NetworkX or similar.

**API:**

```python
import networkx as nx
from .schemas import Deal, Sector, Region, Event, Snippet

class DealGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_deals(self, deals: List[Deal]): ...
    def add_sectors(self, sectors: List[Sector]): ...
    def add_regions(self, regions: List[Region]): ...
    def add_events(self, events: List[Event]): ...
    def add_snippets(self, snippets: List[Snippet]): ...

    def get_deal_neighbors(self, deal_id: str) -> dict:
        """
        Returns neighbors split by type:
        sectors, regions, events, snippets, related deals.
        """
```

Use node attributes:

* `node["type"] in {"deal", "sector", "region", "event", "snippet"}`

Use edge attributes:

* `relation="IN_SECTOR"`, `"IN_REGION"`, `"ADDON_TO"`, `"EXITED_VIA"`, `"DESCRIBED_IN"`.

---

### 5.2 Embeddings & Index

#### 5.2.1 `dealgraph/embeddings/encoder.py`

**Responsibilities:**

* Provide a simple interface to compute text embeddings.

**API:**

```python
class EmbeddingEncoder:
    def __init__(self, model_name: str):
        ...

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return list of embedding vectors."""
```

#### 5.2.2 `dealgraph/embeddings/index.py`

**Responsibilities:**

* Maintain a vector index for deal-level text (deal description + key snippets).

**API:**

```python
class DealEmbeddingIndex:
    def __init__(self, dim: int):
        ...

    def add(self, deal_id: str, vector: List[float]):
        ...

    def search(self, query_vector: List[float], top_k: int = 50) -> List[tuple]:
        """
        Returns list of (deal_id, similarity_score)
        """
```

Start with in-memory brute-force cosine; FAISS can be a later optimization.

---

### 5.3 Retrieval

#### 5.3.1 `dealgraph/retrieval/features.py`

**Responsibilities:**

* Compute **graph-derived features** for a candidate deal given the query context.

**API:**

```python
def compute_graph_features(
    deal_graph: DealGraph,
    deal: Deal,
    query_sectors: List[str],
    query_regions: List[str]
) -> dict:
    """
    Returns a dict of features:
      - sector_match: 0/1
      - region_match: 0/1
      - num_addons: int
      - has_exit: 0/1
      - degree: int
    """
```

#### 5.3.2 `dealgraph/retrieval/graph_search.py`

**Responsibilities:**

* Implement **graph-aware semantic retrieval**.

**API:**

```python
from ..data.schemas import CandidateDeal

def graph_semantic_search(
    query: str,
    encoder: EmbeddingEncoder,
    deal_index: DealEmbeddingIndex,
    deal_graph: DealGraph,
    deals: List[Deal],
    snippets_by_deal: dict,
    top_k: int = 50
) -> List[CandidateDeal]:
    """
    1. Embed query.
    2. Retrieve top_k deal IDs by embedding similarity.
    3. For each candidate deal:
       - Collect representative snippets.
       - Compute graph_features using `compute_graph_features`.
    4. Return CandidateDeal list (unordered).
    """
```

Do **not** apply ML ranking here; just compute similarity + features.

---

### 5.4 Ranking

#### 5.4.1 `dealgraph/ranking/features.py`

**Responsibilities:**

* Turn `CandidateDeal` into a numerical feature vector for the ML ranker.

**API:**

```python
import numpy as np

FEATURE_NAMES = [
    "text_similarity",
    "sector_match",
    "region_match",
    "num_addons",
    "has_exit",
    "degree"
    # extend as needed
]

def candidate_to_features(candidate: CandidateDeal) -> np.ndarray:
    ...
```

#### 5.4.2 `dealgraph/ranking/model.py`

**Responsibilities:**

* Define ML ranker (e.g., gradient boosted regression / classification model).

**API:**

```python
from typing import List
import numpy as np

class DealRanker:
    def __init__(self, model=None):
        self.model = model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        ...

    def predict_scores(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        ...

    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str) -> "DealRanker": ...
```

#### 5.4.3 `dealgraph/ranking/data_gen.py`

**Responsibilities:**

*   Implement "Reverse-Query" generation to create synthetic training data (Option B).
*   Sample deal clusters from `DealGraph`.
*   Prompt LLM to generate queries for those clusters.
*   Generate `(query, candidate_deal, label)` tuples.

**API:**

```python
def generate_synthetic_training_data(
    deal_graph: DealGraph,
    num_samples: int = 1000
) -> List[dict]:
    """
    Returns list of training examples:
    {
      "query": str,
      "candidate_id": str,
      "label": int  # 1 for relevant, 0 for irrelevant
    }
    """
```

#### 5.4.4 `dealgraph/ranking/train.py`

**Responsibilities:**

* Training script using **DealGraph Bench** labels.

**API:**

```python
def train_ranker_from_bench(
    bench_dataset,
    output_path: str
) -> None:
    """
    1. Load bench queries and labeled precedents.
    2. For each query:
       - Run graph_semantic_search to get candidates.
       - Generate features for each candidate.
       - Assign labels (e.g., 1 for relevant, 0 for non-relevant).
    3. Train DealRanker and save to disk.
    """
```

#### 5.4.5 Ranking Tool

High-level API for agent:

```python
from ..data.schemas import CandidateDeal, RankedDeal

def rank_deals(
    query: str,
    candidates: List[CandidateDeal],
    ranker: DealRanker
) -> List[RankedDeal]:
    """
    1. Build feature matrix from candidates.
    2. Use ranker to predict scores.
    3. Sort descending, assign rank.
    """
```

---

### 5.5 Reasoning

#### 5.5.1 Prompt Versioning & Management

**Directory Structure:**

```text
prompts/
  deal_reasoner/
    v1_naive.txt           # Hand-written baseline prompt
    v2_optimized.json      # DSPy MIPRO-optimized prompt
    CHANGELOG.md           # Semantic versioning log (X.Y.Z)
  reverse_query/
    v1_naive.txt
    CHANGELOG.md
```

**Versioning Strategy:**

* Use **semantic versioning** (MAJOR.MINOR.PATCH):
  * MAJOR: Breaking changes to prompt structure or output format
  * MINOR: Improvements that maintain compatibility
  * PATCH: Bug fixes, typo corrections
* Store prompts as files in Git, not hardcoded strings
* Track metadata: model, temperature, max_tokens, performance metrics
* CHANGELOG.md documents each version with rationale and benchmark results

**API:**

```python
from pathlib import Path
from typing import Optional

class PromptRegistry:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
    
    def load_prompt(
        self, 
        prompt_name: str, 
        version: Optional[str] = None
    ) -> dict:
        """
        Load a prompt by name and version.
        If version is None, loads the latest version.
        Returns dict with: content, metadata, version
        """
    
    def list_versions(self, prompt_name: str) -> List[str]:
        """List all versions for a given prompt."""
    
    def get_metadata(self, prompt_name: str, version: str) -> dict:
        """Get metadata for a specific prompt version."""
```

#### 5.5.2 `dealgraph/reasoning/prompts.py`

**Responsibilities:**

* Define naive baseline prompts as constants (for fallback).
* Load optimized prompts via PromptRegistry.

**Example:**

```python
# Naive baseline (v1) - used if optimization hasn't run yet
DEAL_REASONER_NAIVE_PROMPT = """
You are a private-equity deal research assistant. Given a new opportunity description and a set of historical deals, your task is to:
1. Identify which historical deals are the best precedents.
2. Explain why they are precedents.
3. Extract common 'playbook levers' used in these deals.
4. Extract common risk themes.
5. Return a JSON object with fields: precedents, playbook_levers, risk_themes, narrative_summary.

New opportunity:
{query}

Candidate deals:
{deals_block}

Each deal has the format:
- id: <deal_id>
- name: <deal_name>
- description: <one-line description>
- snippets: <key paragraphs from news/case studies>
- sector: <sector>
- region: <region>
- status: <current/realized>
- metadata: <buy-and-build / add-ons / exits information>

Please analyze and respond ONLY with valid JSON.
"""
```

#### 5.5.3 `dealgraph/reasoning/llm_client.py`

**Responsibilities:**

* Wrap calls to the chosen LLM API.
* Configure DSPy LM backend.

**API:**

```python
import dspy

class LLMClient:
    def __init__(self, model_name: str, api_key_env_var: str = "OPENAI_API_KEY"):
        self.model_name = model_name
        # Configure DSPy backend
        self.lm = dspy.LM(model=model_name, api_key=os.getenv(api_key_env_var))
        dspy.configure(lm=self.lm)

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Calls the LLM and parses JSON response.
        Used for non-DSPy calls (e.g., LLM-as-judge).
        """
```

#### 5.5.4 `dealgraph/reasoning/dspy_modules.py`

**Responsibilities:**

* Define DSPy signatures and modules for prompt optimization.

**API:**

```python
import dspy
from typing import List
from ..data.schemas import RankedDeal

class DealReasonerSignature(dspy.Signature):
    """Analyze historical PE deals to identify precedents and extract strategic insights."""
    
    query: str = dspy.InputField(
        desc="New deal opportunity description from user"
    )
    candidate_deals: str = dspy.InputField(
        desc="JSON-formatted list of historical deals with metadata, snippets, and graph features"
    )
    
    precedents: str = dspy.OutputField(
        desc="JSON list of most relevant precedent deals with similarity explanations"
    )
    playbook_levers: str = dspy.OutputField(
        desc="JSON list of common value-creation strategies observed across precedents"
    )
    risk_themes: str = dspy.OutputField(
        desc="JSON list of common risk patterns and mitigation approaches"
    )
    narrative_summary: str = dspy.OutputField(
        desc="Executive summary synthesizing the analysis in 2-3 paragraphs"
    )

class DealReasonerModule(dspy.Module):
    """DSPy module for deal reasoning with MIPRO optimization."""
    
    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(DealReasonerSignature)
    
    def forward(self, query: str, candidate_deals: str):
        result = self.reason(query=query, candidate_deals=candidate_deals)
        return dspy.Prediction(
            precedents=result.precedents,
            playbook_levers=result.playbook_levers,
            risk_themes=result.risk_themes,
            narrative_summary=result.narrative_summary
        )
```

#### 5.5.5 `dealgraph/reasoning/optimizer.py`

**Responsibilities:**

* Implement prompt optimization using DSPy MIPRO.
* Define composite evaluation metric (precision + quality).

**API:**

```python
import dspy
from dspy.teleprompt import MIPRO
from typing import List, Callable
from ..eval.bench_dataset import BenchQuery
from .dspy_modules import DealReasonerModule

class DealReasonerMetric:
    """Composite metric: precision + output quality."""
    
    def __init__(self, llm_judge: LLMClient):
        self.llm_judge = llm_judge
    
    def __call__(self, example: BenchQuery, prediction: dspy.Prediction) -> float:
        """
        Returns score in [0, 1]:
        - 0.4: Precision@3 for precedent selection
        - 0.3: Playbook quality (LLM-as-judge)
        - 0.3: Narrative coherence (LLM-as-judge)
        """
        score = 0.0
        
        # Precision component
        try:
            predicted_ids = self._extract_precedent_ids(prediction.precedents)
            precision = self._precision_at_k(
                predicted_ids, 
                example.relevant_deal_ids, 
                k=3
            )
            score += 0.4 * precision
        except:
            pass  # Invalid JSON = 0 precision score
        
        # Quality components (LLM-as-judge)
        playbook_score = self._judge_playbook_quality(
            prediction.playbook_levers
        )
        narrative_score = self._judge_narrative_coherence(
            prediction.narrative_summary
        )
        
        score += 0.3 * playbook_score
        score += 0.3 * narrative_score
        
        return score
    
    def _extract_precedent_ids(self, precedents_json: str) -> List[str]:
        """Parse JSON and extract deal IDs."""
    
    def _precision_at_k(
        self, 
        predicted: List[str], 
        relevant: List[str], 
        k: int
    ) -> float:
        """Standard precision@k metric."""
    
    def _judge_playbook_quality(self, playbook_json: str) -> float:
        """
        LLM-as-judge: Are playbook levers specific, actionable, 
        and grounded in the deals?
        Returns score in [0, 1].
        """
    
    def _judge_narrative_coherence(self, narrative: str) -> float:
        """
        LLM-as-judge: Is narrative coherent, executive-appropriate,
        and well-structured?
        Returns score in [0, 1].
        """

def optimize_deal_reasoner(
    train_examples: List[BenchQuery],
    metric: DealReasonerMetric,
    num_candidates: int = 10,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3,
    output_path: str = "prompts/deal_reasoner/v2_optimized.json"
) -> DealReasonerModule:
    """
    Run DSPy MIPRO optimization on the deal reasoner.
    
    Args:
        train_examples: DealGraph Bench queries with labels
        metric: Composite evaluation metric
        num_candidates: Number of prompt candidates to generate
        max_bootstrapped_demos: Max bootstrapped examples per prompt
        max_labeled_demos: Max labeled examples per prompt
        output_path: Where to save optimized module
    
    Returns:
        Optimized DealReasonerModule
    """
    # Initialize naive module
    reasoner = DealReasonerModule()
    
    # MIPRO optimizer (meta-prompting, no few-shot examples)
    optimizer = MIPRO(
        metric=metric,
        num_candidates=num_candidates,
        init_temperature=1.0
    )
    
    # Run optimization
    optimized_reasoner = optimizer.compile(
        reasoner,
        trainset=train_examples,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        requires_permission_to_run=False
    )
    
    # Save optimized module
    optimized_reasoner.save(output_path)
    
    return optimized_reasoner
```

#### 5.5.6 `dealgraph/reasoning/reasoner.py`

**Responsibilities:**

* Implement `deal_reasoner()` using optimized DSPy module.
* Fall back to naive prompt if optimization hasn't run.
* Fail loudly on errors (no silent fallbacks).

**API:**

```python
from typing import List
from pathlib import Path
import dspy
from ..data.schemas import RankedDeal, DealReasoningOutput, Precedent
from .dspy_modules import DealReasonerModule
from .prompts import DEAL_REASONER_NAIVE_PROMPT

class DealReasonerError(Exception):
    """Raised when deal reasoning fails."""
    pass

def deal_reasoner(
    query: str,
    ranked_deals: List[RankedDeal],
    max_deals: int = 10,
    prompt_version: str = "latest"
) -> DealReasoningOutput:
    """
    Analyze ranked deals and extract precedents, playbooks, risks.
    
    Args:
        query: User's deal query
        ranked_deals: Ranked candidate deals from retrieval
        max_deals: Max deals to include in reasoning
        prompt_version: Prompt version to use ("latest", "v1", "v2", etc.)
    
    Returns:
        DealReasoningOutput with structured analysis
    
    Raises:
        DealReasonerError: If reasoning fails (malformed output, API error, etc.)
    """
    # Take top K deals
    top_deals = ranked_deals[:max_deals]
    
    # Format deals as JSON string
    deals_block = _format_deals_for_prompt(top_deals)
    
    # Load optimized module or use naive prompt
    try:
        if prompt_version == "latest":
            reasoner = DealReasonerModule()
            reasoner.load("prompts/deal_reasoner/v2_optimized.json")
        elif prompt_version == "v1":
            # Use naive prompt
            reasoner = None
        else:
            reasoner = DealReasonerModule()
            reasoner.load(f"prompts/deal_reasoner/{prompt_version}.json")
    except FileNotFoundError:
        # Optimized prompt doesn't exist yet, use naive
        reasoner = None
    
    # Execute reasoning
    try:
        if reasoner:
            # Use DSPy module
            result = reasoner(query=query, candidate_deals=deals_block)
            output = _parse_dspy_output(result)
        else:
            # Use naive prompt
            output = _execute_naive_prompt(query, deals_block)
        
        return output
    
    except Exception as e:
        raise DealReasonerError(
            f"Deal reasoning failed for query '{query[:50]}...': {str(e)}"
        ) from e

def _format_deals_for_prompt(deals: List[RankedDeal]) -> str:
    """Convert ranked deals to JSON string for prompt."""

def _parse_dspy_output(result: dspy.Prediction) -> DealReasoningOutput:
    """Parse DSPy output into DealReasoningOutput schema."""

def _execute_naive_prompt(query: str, deals_block: str) -> DealReasoningOutput:
    """Execute naive baseline prompt (non-DSPy path)."""
```

---

### 5.6 Agent

#### 5.6.1 `dealgraph/agent/tools.py`

**Responsibilities:**

* Thin wrappers so agent logic is clean.

**API:**

```python
def tool_graph_semantic_search(query: str) -> List[CandidateDeal]: ...
def tool_rank_deals(query: str, candidates: List[CandidateDeal]) -> List[RankedDeal]: ...
def tool_deal_reasoner(query: str, ranked_deals: List[RankedDeal]) -> DealReasoningOutput: ...
```

#### 5.6.2 `dealgraph/agent/orchestrator.py`

**Responsibilities:**

* Implement `run_agent()`.

**API:**

```python
from ..data.schemas import DealReasoningOutput
from pydantic import BaseModel

class AgentLog(BaseModel):
    query: str
    tool_calls: list  # each entry: {tool_name, inputs_summary, outputs_summary}
    reasoning_output: DealReasoningOutput

def run_agent(query: str) -> AgentLog:
    """
    Orchestration:
    1. tool_graph_semantic_search
    2. tool_rank_deals
    3. tool_deal_reasoner
    4. Return AgentLog
    """
```

---

## 6. Evaluation: DealGraph Bench

### 6.1 Dataset

Implement in `dealgraph/eval/bench_dataset.py`.

Represent as:

```python
from pydantic import BaseModel
from typing import List

class BenchQuery(BaseModel):
    id: str
    text: str
    relevant_deal_ids: List[str]  # known precedents
```

Store in `data/bench/bench_queries.json`.

Guidelines:

* ~20–30 high-quality queries.
* Each query maps to 1–3 clearly related deals.
* Include some “hard negatives” (different sectors/regions/models).

### 6.2 Metrics

`dealgraph/eval/metrics.py`:

```python
def recall_at_k(
    ranked_deal_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float: ...

def ndcg_at_k(
    ranked_deal_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float: ...
```

### 6.3 Comparison Script

`dealgraph/eval/compare_ranking.py`:

* For each `BenchQuery`:

  * Run `graph_semantic_search` to get candidates.
  * **Baseline** ranking: sort by `text_similarity`.
  * **ML ranker** ranking: use `rank_deals` with trained model.
* Compute average Recall@k and NDCG@k for both.
* Print a simple comparison table.

---

## 7. CLI

`dealgraph/cli/main.py` (e.g., using Typer):

```python
import typer
from ..agent.orchestrator import run_agent

app = typer.Typer()

@app.command()
def query(q: str):
    """
    Run agent on a free-text query and print JSON + narrative summary.
    """
    log = run_agent(q)
    print(log.reasoning_output.model_dump_json(indent=2))
    print("\n--- Narrative ---\n")
    print(log.reasoning_output.narrative_summary)

if __name__ == "__main__":
    app()
```

Expose as console script in `pyproject.toml`:

```toml
[project.scripts]
dealgraph-agent = "dealgraph.cli.main:app"
```

---

## 8. Testing

Use pytest. Suggested tests:

* `test_data_ingest.py` — loading and validating schemas.
* `test_graph_search.py` — basic retrieval sanity (returns candidates, features).
* `test_ranking_model.py` — feature shapes, score ordering, basic training.
* `test_reasoner.py` — mock LLM client, JSON parsing, output fields present.
* `test_prompt_optimization.py` — DSPy module loading, metric computation, optimization workflow.
* `test_agent_orchestrator.py` — end-to-end call using mocks, correct tool sequence.

Use mocks/stubs for external APIs (LLM, embeddings) for reproducibility.

---

## 9. Implementation Priorities

Recommended order of implementation:

1. **Data & schemas** (`data/schemas.py`, `ingest.py`).
2. **Graph builder & embedding index** (`graph_builder.py`, `encoder.py`, `index.py`).
3. **Graph-aware retrieval** (`graph_search.py`, `retrieval/features.py`).
4. **Benchmark & metrics** (`eval/bench_dataset.py`, `eval/metrics.py`).
5. **Ranking model** (`ranking/features.py`, `ranking/model.py`, `ranking/data_gen.py`, `ranking/train.py`, `rank_deals()`).
6. **Reasoning layer (naive baseline)** (`reasoning/prompts.py`, `reasoning/llm_client.py`, `reasoning/dspy_modules.py`, `reasoning/reasoner.py`).
7. **Prompt optimization** (`reasoning/optimizer.py`, run optimization on DealGraph Bench, save optimized prompts).
8. **Agent orchestrator** (`agent/tools.py`, `agent/orchestrator.py`) + CLI.
9. **Tests & cleanup** (all `tests/`).

---

## 10. Prompt Optimization Workflow

### Initial Setup

1. Implement naive baseline prompt (`prompts/deal_reasoner/v1_naive.txt`)
2. Build DealGraph Bench with 20-30 labeled queries
3. Implement composite metric (precision + LLM-as-judge quality)

### Optimization Process

```bash
# Run DSPy MIPRO optimization
python -m dealgraph.reasoning.optimizer \
  --bench-path data/bench/bench_queries.json \
  --output prompts/deal_reasoner/v2_optimized.json \
  --num-candidates 10 \
  --max-iterations 50

# Evaluate optimized prompt
python -m dealgraph.eval.compare_prompts \
  --baseline v1_naive.txt \
  --optimized v2_optimized.json \
  --output results/prompt_comparison.json

# Update CHANGELOG
echo "## v2.0.0 - YYYY-MM-DD
- MAJOR: DSPy MIPRO optimization
- Improved precision@3 from X% to Y%
- Improved narrative coherence score from A to B
- Model: gpt-4o, temp=0.7" >> prompts/deal_reasoner/CHANGELOG.md
```

### Re-optimization Triggers

Re-run optimization when:
* DealGraph Bench grows significantly (>50% more examples)
* Switching to a different LLM model
* Performance degrades on production queries
* New output requirements emerge

---

## 11. Dependencies

Add to `requirements.txt`:

```text
# Core
pydantic>=2.0
networkx>=3.0
numpy>=1.24
pandas>=2.0

# Embeddings & Vector Search
openai>=1.0  # or sentence-transformers
faiss-cpu>=1.7  # optional, can use numpy for v1

# ML Ranking
scikit-learn>=1.3
xgboost>=2.0  # or lightgbm

# Prompt Optimization
dspy-ai>=2.5

# CLI
typer>=0.9
rich>=13.0  # for pretty CLI output

# Testing
pytest>=7.0
pytest-mock>=3.0
```
