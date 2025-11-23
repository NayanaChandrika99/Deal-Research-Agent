# Execution Plans

This file contains the active execution plans for complex tasks.
Copy the template below for each new major feature or refactor.

## Template

### [Plan-ID] Title of Task

**Goal**: One sentence summary of what we are achieving.

**Context**: Why are we doing this? Link to `SPECIFICATION.md` or `ARCHITECTURE.md`.

**Proposed Changes**:
- [ ] Step 1: Description
- [ ] Step 2: Description

**Verification Plan**:
- [ ] Unit Test: `tests/path/to/test.py`
- [ ] Manual Verification: Command to run

---

## ExecPlan 1.1: Data Schemas

**Goal**: Implement Pydantic data models for all DealGraph entities and reasoning structures.

**Context**: Foundation layer for the entire system - defines the data structures that all other components will use. All other ExecPlans depend on having these schemas defined first. Referenced in SPECIFICATION.md §4 ("Data Model & Schemas") and ARCHITECTURE.md §3 ("Data Model").

**Status**: ✅ **COMPLETED**

**Proposed Changes**:
- [x] Create `dealgraph/data/schemas.py` with Pydantic models:
  - [x] `Sector`, `Region` - taxonomy nodes
  - [x] `Event`, `EventType` - time-based actions (addon, exit)
  - [x] `Snippet` - textual evidence
  - [x] `Deal` - portfolio company investment
  - [x] `CandidateDeal` - search result with similarity and features
  - [x] `RankedDeal` - candidate with ML score and rank
  - [x] `Precedent`, `DealReasoningOutput` - reasoning analysis outputs
  - [x] `DealDataset` - complete data bundle
- [x] Add comprehensive docstrings explaining each model's purpose
- [x] Include type aliases for convenience (`DealList`, etc.)
- [x] Create unit tests validating schema constraints and relationships

**Verification Plan**:
- [x] Unit Test: `tests/test_data_schemas.py` - validate all model instantiation, field validation, and relationships (18/18 tests passed)
- [x] Manual Verification: Run `python -c "from dealgraph.data.schemas import Deal, Sector; print('Schemas working')"` to verify imports
- [x] Integration: Ensure schemas integrate properly with existing `EmbeddingEncoder` from `dealgraph/embeddings/encoder.py`

**References**:
- SPECIFICATION.md § 4 ("Data Model & Schemas")
- SPECIFICATION.md § 4.1 ("Core entities") 
- ARCHITECTURE.md § 3 ("Data Model")

**Estimated Effort**: 2-3 hours
**Prerequisites**: None (can start immediately)
**Next ExecPlan**: 1.2 (Data Ingestion)

---

## ExecPlan 1.2: Data Ingestion

**Goal**: Implement data loading functions that ingest raw deal data and validate referential integrity.

**Context**: The ingestion layer is responsible for loading raw data files (JSON/CSV/Markdown) and normalizing them into Pydantic models. This layer validates the data and ensures referential integrity before it's used to build the graph. Referenced in SPECIFICATION.md §5.1.1 (`dealgraph/data/ingest.py`).

**Status**: ✅ **COMPLETED**

**Proposed Changes**:
- [x] Create `dealgraph/data/ingest.py` with loading functions:
  - [x] `load_sectors(path: str) -> SectorList` - load sectors from JSON/CSV
  - [x] `load_regions(path: str) -> RegionList` - load regions from JSON/CSV
  - [x] `load_deals(path: str) -> DealList` - load deals from JSON/CSV
  - [x] `load_events(path: str) -> EventList` - load events from JSON/CSV
  - [x] `load_snippets(path: str) -> SnippetList` - load snippets from JSON/CSV
  - [x] `load_all(base_path: str) -> DealDataset` - load everything together
- [x] Implement validation:
  - [x] Check for missing IDs in all entities
  - [x] Validate referential integrity (deal.sector_id must exist in sectors)
  - [x] Ensure all deal relationships are valid
- [x] Create sample synthetic data in `data/raw/`:
  - [x] `sectors.json` - sample sector taxonomy (5 sectors)
  - [x] `regions.json` - sample region definitions (4 regions)
  - [x] `deals.json` - sample deal portfolio (12 deals)
  - [x] `events.json` - sample add-on and exit events (8 events)
  - [x] `snippets.json` - sample textual evidence (12 snippets)
- [x] Create unit tests for loading logic and validation

**Verification Plan**:
- [x] Unit Test: `tests/test_data_ingest.py` - validate all loading functions work correctly (21/21 tests passed)
- [x] Integration Test: Load sample data and verify `DealDataset` creation
- [x] Manual Verification: Run `python -c "from dealgraph.data.ingest import load_all; dataset = load_all('data/raw'); print(f'Loaded {len(dataset.deals)} deals')"`
- [x] Validation Test: Test that invalid data (missing IDs, broken references) raises appropriate errors

**References**:
- SPECIFICATION.md § 5.1.1 (`dealgraph/data/ingest.py`)
- SPECIFICATION.md § 3 (Repository Layout - `data/raw/`)

**Estimated Effort**: 2-3 hours
**Prerequisites**: ExecPlan 1.1 (Data Schemas) - ✅ COMPLETED, ExecPlan 1.2 (Data Ingestion) - ✅ COMPLETED
**Next ExecPlan**: 1.3 (Graph Builder)

---

## ExecPlan 1.3: Graph Builder

**Goal**: Build NetworkX MultiDiGraph from validated deal data with proper relationships and neighbor queries.

**Context**: The graph builder creates the core DealGraph structure that enables graph-aware search and retrieval. It transforms the ingested deal data into a NetworkX graph with typed edges connecting deals, sectors, regions, events, and snippets. Referenced in SPECIFICATION.md §5.1.2 (`dealgraph/data/graph_builder.py`).

**Status**: ✅ **COMPLETED**

**Proposed Changes**:
- [x] Create `dealgraph/data/graph_builder.py` with `DealGraph` class:
  - [x] `add_deals(deals: DealList)` - add deal nodes with attributes
  - [x] `add_sectors(sectors: SectorList)` - add sector nodes  
  - [x] `add_regions(regions: RegionList)` - add region nodes
  - [x] `add_events(events: EventList)` - add event nodes and relationships
  - [x] `add_snippets(snippets: SnippetList)` - add snippet nodes and text edges
- [x] Implement `get_deal_neighbors(deal_id: str) -> dict`:
  - [x] Returns neighbors split by type: sectors, regions, events, snippets, related deals
- [x] NetworkX MultiDiGraph with proper edge types:
  - [x] `IN_SECTOR` edge: Deal -> Sector
  - [x] `IN_REGION` edge: Deal -> Region  
  - [x] `ADDON_TO` edge: Deal -> Deal (platform to addon)
  - [x] `EXITED_VIA` edge: Deal -> Event
  - [x] `DESCRIBED_IN` edge: Deal -> Snippet
- [x] Create integration test verifying graph connectivity
- [x] Update data module exports to include graph builder

**Verification Plan**:
- [x] Unit Test: `tests/test_graph_builder.py` - validate graph construction and neighbor queries (19/19 tests passed)
- [x] Integration Test: Load data via `load_all()` → build graph → verify relationships
- [x] Manual Verification: Run `python -c "from dealgraph.data.graph_builder import DealGraph; print('Graph builder working')"`
- [x] Connectivity Test: Verify specific deal relationships exist in built graph

**References**:
- SPECIFICATION.md § 5.1.2 (`dealgraph/data/graph_builder.py`)
- ARCHITECTURE.md § 3 ("Graph Schema")
- ARCHITECTURE.md § 7 ADR-001 ("NetworkX for Graph Storage")

**Estimated Effort**: 2-3 hours
**Prerequisites**: ExecPlan 1.2 (Data Ingestion) - ✅ COMPLETED
**Next ExecPlan**: 2.1 (Embeddings Index)

---

## ExecPlan 2.1: Embeddings Index

**Goal**: Implement vector storage and search capabilities for deal embeddings using the existing EmbeddingEncoder.

**Context**: The embeddings layer enables semantic search over deal descriptions and snippets. Building on the completed `EmbeddingEncoder`, we need to implement `DealEmbeddingIndex` for efficient vector storage and similarity search. This enables the foundation for semantic retrieval in the next phase. Referenced in SPECIFICATION.md §5.2.2 (`dealgraph/embeddings/index.py`).

**Status**: 🔄 **IN PROGRESS**

**Proposed Changes**:
- [ ] Create `dealgraph/embeddings/index.py` with `DealEmbeddingIndex` class:
  - [ ] `add(deal_id: str, vector: np.ndarray)` - store deal embeddings
  - [ ] `search(query_vector: np.ndarray, top_k: int = 50) -> List[tuple]` - similarity search
  - [ ] `get_vector(deal_id: str) -> Optional[np.ndarray]` - retrieve stored vectors
  - [ ] `remove(deal_id: str)` - delete embeddings
- [ ] Implement vector storage with efficient similarity computation:
  - [ ] Start with in-memory numpy arrays (can optimize to FAISS later)
  - [ ] Use cosine similarity for semantic matching
  - [ ] Support batch operations for performance
- [ ] Integration with `EmbeddingEncoder`:
  - [ ] Auto-detect embedding dimension from encoder
  - [ ] Use same model configuration for consistency
- [ ] Create embeddings module exports in `embeddings/__init__.py`
- [ ] Update embeddings config to include index settings

**Verification Plan**:
- [ ] Unit Test: `tests/test_embeddings_index.py` - validate vector storage and search
- [ ] Integration Test: Encode sample text → store in index → search and verify results
- [ ] Performance Test: Verify search performance with realistic dataset size
- [ ] Manual Verification: Run `python -c "from dealgraph.embeddings import DealEmbeddingIndex; print('Index working')"`

**References**:
- SPECIFICATION.md § 5.2.2 (`dealgraph/embeddings/index.py`)
- SETUP.md ("OpenAI Embeddings API")
- Existing `dealgraph/embeddings/encoder.py` - ✅ COMPLETED

**Estimated Effort**: 2-3 hours
**Prerequisites**: ExecPlan 1.3 (Graph Builder) - ✅ COMPLETED
**Next ExecPlan**: 2.2 (Graph Search Logic)

---
