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

## Future Versions

### v2.0.0 - DSPy MIPRO Optimization
**Planned Features**:
- DSPy MIPRO-optimized prompts
- Automated prompt improvement
- Performance metrics integration
- Composite evaluation scoring

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
