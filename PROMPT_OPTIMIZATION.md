# Prompt Optimization Guide

## Overview

This document describes the DSPy-based prompt optimization strategy for the DealGraph Agent, focusing on the Deal Reasoner component.

## Quick Start

### 1. Initial Setup

```bash
# Install dependencies
pip install dspy-ai>=2.5

# Create prompt directory structure
mkdir -p prompts/deal_reasoner
```

### 2. Create Naive Baseline

Write a hand-crafted prompt and save to `prompts/deal_reasoner/v1_naive.txt`:

```
You are a private-equity deal research assistant...
[See SPECIFICATION.md § 5.5.2 for full template]
```

### 3. Build Evaluation Dataset

Create DealGraph Bench with 20-30 labeled queries:

```python
# data/bench/bench_queries.json
[
  {
    "id": "query_001",
    "text": "US industrial distribution roll-up with multiple add-ons",
    "relevant_deal_ids": ["deal_042", "deal_089", "deal_156"]
  },
  ...
]
```

### 4. Run Optimization

```bash
python -m dealgraph.reasoning.optimizer \
  --bench-path data/bench/bench_queries.json \
  --output prompts/deal_reasoner/v2_optimized.json \
  --num-candidates 10
```

This will:
- Generate 10 prompt variants using MIPRO
- Evaluate each on the benchmark using the composite metric
- Save the best-performing prompt

### 5. Evaluate Results

```bash
python -m dealgraph.eval.compare_prompts \
  --baseline v1_naive.txt \
  --optimized v2_optimized.json
```

Expected output:
```
Prompt Comparison Results
=========================
Metric                    v1 (naive)    v2 (optimized)    Δ
--------                  ----------    --------------    ----
Precision@3               0.42          0.68              +62%
Playbook Quality          0.55          0.72              +31%
Narrative Coherence       0.61          0.78              +28%
--------                  ----------    --------------    ----
Composite Score           0.52          0.73              +40%
```

## Composite Metric Design

The optimization metric balances precision and quality:

```python
score = 0.4 * precision@3 + 0.3 * playbook_quality + 0.3 * narrative_coherence
```

### Component 1: Precision@3 (40%)

**Objective measurement**: Did the model select the correct precedent deals?

```python
def precision_at_k(predicted_ids, relevant_ids, k=3):
    top_k = predicted_ids[:k]
    correct = len(set(top_k) & set(relevant_ids))
    return correct / k
```

### Component 2: Playbook Quality (30%)

**LLM-as-judge**: Are the extracted playbook levers specific, actionable, and grounded?

Judge prompt:
```
Evaluate the following playbook levers on a scale of 0-1:
- Specificity: Are they concrete vs. generic?
- Actionability: Can they guide actual decisions?
- Grounding: Are they supported by the deal evidence?

Playbook levers: {playbook_json}
Deal context: {deals_summary}

Return only a float score between 0 and 1.
```

### Component 3: Narrative Coherence (30%)

**LLM-as-judge**: Is the narrative well-structured and executive-appropriate?

Judge prompt:
```
Evaluate the following narrative summary on a scale of 0-1:
- Structure: Clear introduction, body, conclusion?
- Coherence: Logical flow between ideas?
- Tone: Appropriate for executive audience?

Narrative: {narrative_text}

Return only a float score between 0 and 1.
```

## DSPy Module Structure

### Signature Definition

```python
class DealReasonerSignature(dspy.Signature):
    """Analyze historical PE deals to identify precedents and extract strategic insights."""
    
    query: str = dspy.InputField(
        desc="New deal opportunity description from user"
    )
    candidate_deals: str = dspy.InputField(
        desc="JSON-formatted list of historical deals with metadata"
    )
    
    precedents: str = dspy.OutputField(
        desc="JSON list of most relevant precedent deals"
    )
    playbook_levers: str = dspy.OutputField(
        desc="JSON list of common value-creation strategies"
    )
    risk_themes: str = dspy.OutputField(
        desc="JSON list of common risk patterns"
    )
    narrative_summary: str = dspy.OutputField(
        desc="Executive summary synthesizing the analysis"
    )
```

### Module Implementation

```python
class DealReasonerModule(dspy.Module):
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

## MIPRO Optimizer Configuration

```python
from dspy.teleprompt import MIPRO

optimizer = MIPRO(
    metric=composite_metric,
    num_candidates=10,        # Generate 10 prompt variants
    init_temperature=1.0      # Creativity in prompt generation
)

optimized = optimizer.compile(
    reasoner,
    trainset=bench_examples,
    max_bootstrapped_demos=3,  # Max bootstrapped examples
    max_labeled_demos=3,       # Max labeled examples
    requires_permission_to_run=False
)
```

**Key parameters**:
- `num_candidates`: More = better optimization, but more LLM calls (cost)
- `init_temperature`: Higher = more creative variants, lower = safer iterations
- `max_*_demos`: MIPRO uses meta-prompting, so these stay low (unlike BootstrapFewShot)

## Versioning Workflow

### CHANGELOG.md Format

```markdown
# Deal Reasoner Prompt Changelog

## v2.0.0 - 2024-01-15

**Type**: MAJOR - DSPy MIPRO optimization

**Metrics**:
- Precision@3: 0.42 → 0.68 (+62%)
- Playbook Quality: 0.55 → 0.72 (+31%)
- Narrative Coherence: 0.61 → 0.78 (+28%)
- Composite Score: 0.52 → 0.73 (+40%)

**Configuration**:
- Model: gpt-4o
- Temperature: 0.7
- Max tokens: 2000
- Optimizer: MIPRO (10 candidates, 50 iterations)

**Benchmark**: DealGraph Bench v1 (25 queries)

**Notes**: Significant improvement in precedent selection accuracy.

---

## v1.0.0 - 2024-01-01

**Type**: Initial naive baseline

**Metrics**:
- Precision@3: 0.42
- Playbook Quality: 0.55
- Narrative Coherence: 0.61
- Composite Score: 0.52

**Configuration**:
- Model: gpt-4o
- Temperature: 0.7
- Max tokens: 2000

**Notes**: Hand-written prompt, no optimization.
```

## Re-optimization Triggers

Run optimization again when:

1. **Benchmark growth**: DealGraph Bench grows by >50% (e.g., 25 → 40 queries)
2. **Model change**: Switching LLM providers or model versions
3. **Performance degradation**: Production metrics drop below threshold
4. **New requirements**: Output format or quality requirements change

## Cost Estimation

Typical MIPRO optimization run:
- **Candidates**: 10 prompt variants
- **Iterations**: ~50 evaluations per candidate
- **Total LLM calls**: ~500 calls
- **Estimated cost** (GPT-4o): $5-10 per optimization run

This is a **one-time cost** that yields persistent improvements.

## Integration with Production

### Loading Optimized Prompts

```python
from dealgraph.reasoning.reasoner import deal_reasoner

# Automatically loads latest optimized version
result = deal_reasoner(
    query="US healthcare roll-up",
    ranked_deals=candidates,
    prompt_version="latest"  # or "v1", "v2", etc.
)
```

### Fallback Behavior

```python
try:
    # Try to load optimized prompt
    reasoner = DealReasonerModule()
    reasoner.load("prompts/deal_reasoner/v2_optimized.json")
except FileNotFoundError:
    # Fall back to naive baseline
    reasoner = None  # Triggers naive prompt path
```

### Error Handling

**Philosophy**: Fail loudly, no silent fallbacks

```python
try:
    result = reasoner(query=query, candidate_deals=deals_json)
except Exception as e:
    raise DealReasonerError(
        f"Deal reasoning failed: {str(e)}"
    ) from e
```

This ensures:
- Prompt errors are caught immediately
- No degraded silent failures in production
- Clear debugging signals

## Best Practices

### 1. Start Simple

Begin with a clear, well-structured naive prompt. DSPy optimizes from this baseline.

### 2. Quality Over Quantity

20-30 high-quality benchmark examples > 100 low-quality examples.

### 3. Balanced Metrics

Don't over-optimize for precision at the expense of quality. The 40/30/30 split balances both.

### 4. Version Everything

Track not just prompts, but also:
- Model configuration (name, temperature, max_tokens)
- Benchmark version
- Metric scores
- Optimization parameters

### 5. A/B Test in Production

Before fully rolling out v2, run A/B tests:
```python
if random.random() < 0.5:
    result = deal_reasoner(query, deals, prompt_version="v1")
else:
    result = deal_reasoner(query, deals, prompt_version="v2")
```

Track real-world performance before committing.

## Troubleshooting

### Issue: Optimization produces worse results

**Cause**: Metric doesn't align with true quality

**Fix**: Refine the composite metric. Add more weight to the failing component.

### Issue: Optimized prompt produces invalid JSON

**Cause**: DSPy didn't learn output format constraints

**Fix**: Add JSON validation to the metric with heavy penalty:
```python
def metric(example, prediction):
    try:
        json.loads(prediction.precedents)
    except:
        return 0.0  # Harsh penalty for invalid JSON
    # ... rest of metric
```

### Issue: Optimization is too expensive

**Cause**: Too many candidates or iterations

**Fix**: Reduce `num_candidates` from 10 to 5, or set iteration limits:
```python
optimizer = MIPRO(
    metric=metric,
    num_candidates=5,  # Reduced
    max_iterations=30  # Add limit
)
```

## References

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [MIPRO Paper](https://arxiv.org/abs/2406.11695)
- [Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)

