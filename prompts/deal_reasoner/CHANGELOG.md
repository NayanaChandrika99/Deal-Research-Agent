# Deal Reasoning Prompt Changelog

All changes to deal reasoning prompts are documented here with semantic versioning.

## v1.0.0 - 2024-11-23

**Type**: MAJOR
**Baseline**: Naive hand-written prompt

### Added
- Initial naive baseline prompt for deal reasoning
- Focus on identifying precedents, playbook levers, and risk themes
- JSON output format with structured analysis
- Support for multi-deal analysis with similarity explanations

### Prompt Structure
- System prompt defining PE research assistant role
- User prompt with opportunity and candidate deals
- JSON response schema with required fields
- Instructions for actionable insight extraction

### Performance
- Designed for Cerebras Llama 3.1 8B model
- Temperature: 0.3 for consistent analysis
- Max tokens: 2500 for detailed responses

### Future Optimizations
- DSPy MIPRO optimization (planned for v2.0.0)
- A/B testing framework
- Performance metrics tracking
- Multi-model evaluation

---

## v2.0.0 - 2025-02-14

**Type**: MAJOR — DSPy MIPRO optimization (`python -m dealgraph.reasoning.optimizer`)

### Metrics (DealGraph Bench v1, 20 labeled queries)
- Precision@3: 0.44 → 0.63 (+0.19)
- Playbook Quality: 0.52 → 0.71 (+0.19)
- Narrative Coherence: 0.58 → 0.76 (+0.18)
- Composite Score: 0.51 → 0.70 (+0.19)

### Configuration
- Model: `llama3.1-8b`
- Optimization temperature: 0.1
- Reasoning temperature: 0.3
- Candidates: 10, Max evaluations: 100
- Dataset: `data/bench/bench_queries.json`

### Artifacts
- `prompts/deal_reasoner/v2_optimized.json`
- `prompts/deal_reasoner/v2_optimized_module.json`
- Optimization logs via `results/prompt_comparison.json`

---

## Future Versions

### v3.0.0 - Advanced Reasoning
**Planned Features**:
- Multi-step reasoning chains
- Risk quantification
- Market timing analysis
- Sector-specific templates

---

## Version History Summary

| Version | Type | Description | Date |
|---------|------|-------------|------|
| v1.0.0 | MAJOR | Initial naive baseline | 2024-11-23 |
| v2.0.0 | MAJOR | DSPy optimized prompt | 2025-02-14 |

## Prompt Evaluation Metrics

Current metrics tracked:
- Precision@3 for precedent selection
- Playbook quality (LLM-as-judge)
- Narrative coherence (LLM-as-judge)
- Composite score (40% precision + 30% playbook + 30% narrative)

## Model Compatibility

- **Primary**: Cerebras Llama 3.1 8B
- **Alternative**: OpenAI GPT-4 (for evaluation)
- **Fallback**: Any OpenAI-compatible API

## Usage Guidelines

1. Always use the latest version unless specific version required
2. Log prompt version used for each reasoning request
3. Track performance metrics for version comparison
4. A/B test new versions before production deployment

## Contact

For questions about prompt versions or optimization, refer to the implementation team.
