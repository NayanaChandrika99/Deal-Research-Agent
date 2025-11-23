### [ExecPlan 3.1] Reasoning Module (Naive Baseline)

**Goal**: Implement the reasoning module with naive baseline prompts for deal analysis and insight generation.

**Context**: Building on the completed retrieval system (data, graph, embeddings, search), we now create the reasoning layer that uses LLMs to analyze search results and generate structured insights. This implements the baseline (non-optimized) prompt system that will later be enhanced with DSPy. Referenced in SPECIFICATION.md §5.5.2.

**Proposed Changes**:
- [ ] Create `dealgraph/reasoning/prompts.py` with prompt management:
  - [ ] `PromptRegistry` class for loading versioned prompts from files
  - [ ] `DEAL_REASONER_NAIVE_PROMPT` baseline template for deal analysis
  - [ ] Support for prompt versioning and metadata
  - [ ] Fallback mechanisms for missing optimized prompts
- [ ] Create `dealgraph/reasoning/llm_client.py` with LLM integration:
  - [ ] `LLMClient` wrapper for Cerebras API with DSPy integration
  - [ ] JSON response parsing and validation
  - [ ] Error handling and retry logic
  - [ ] Support for both naive and optimized prompt paths
- [ ] Create `dealgraph/reasoning/reasoner.py` with baseline reasoning:
  - [ ] `deal_reasoner()` function using naive prompts
  - [ ] Parse LLM responses into `DealReasoningOutput` schema
  - [ ] Integration with retrieval results
  - [ ] Error handling for malformed responses
- [ ] Create baseline prompt files:
  - [ ] `prompts/deal_reasoner/v1_naive.txt` - hand-written baseline prompt
  - [ ] `prompts/deal_reasoner/CHANGELOG.md` - version tracking
- [ ] Create reasoning module exports and integration
- [ ] Update agent tools to include reasoning capabilities

**Verification Plan**:
- [ ] Unit Test: `tests/test_reasoner.py` - validate reasoning functionality
- [ ] Integration Test: Search → Reasoning pipeline with sample data
- [ ] JSON Validation Test: Ensure LLM responses parse correctly
- [ ] Manual Verification: Run complete search + reasoning workflow
- [ ] Baseline Quality Test: Evaluate reasoning output quality

**References**:
- SPECIFICATION.md § 5.5.2 (`dealgraph/reasoning/prompts.py`)
- SPECIFICATION.md § 5.5.3 (`dealgraph/reasoning/llm_client.py`)
- SPECIFICATION.md § 5.5.6 (`dealgraph/reasoning/reasoner.py`)
- PROMPT_OPTIMIZATION.md § "Start Simple"
- Existing components: ✅ COMPLETED
  - Retrieval system (graph search, ranking)
  - Agent tools and integration

**Estimated Effort**: 3-4 hours
**Prerequisites**: 
- ExecPlan 2.3 (End-to-End Retrieval) - ✅ COMPLETED
- ExecPlan 2.2 (Graph Search Logic) - ✅ COMPLETED
**Next ExecPlan**: 3.2 (DealGraph Bench & Metrics)

**Status**: ✅ **COMPLETED**

---
