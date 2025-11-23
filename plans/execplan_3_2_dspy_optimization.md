# ExecPlan 3.2 - DSPy Optimization

## Objective
Implement DSPy MIPRO-based prompt optimization to improve reasoning quality over the naive baseline from ExecPlan 3.1.

## Context
**Previous ExecPlans**: 
- ExecPlan 3.1 (Reasoning Module) - ✅ COMPLETED
- Established naive baseline prompt with 18/18 tests passing
- Created prompt versioning infrastructure

**Next ExecPlan**: 3.3 (Performance Testing & Validation)

## Implementation Details

### Component 1: DSPy Framework Setup
**Files to Create:**
- `dealgraph/reasoning/dspy/` - DSPy optimization module
- `tests/test_dspy_optimization.py` - DSPy test suite

**Requirements:**
- Install DSPy framework and dependencies
- Set up optimization pipeline with MIPRO algorithm
- Configure evaluation metrics (precision, playbook quality, narrative coherence)
- Implement automated prompt improvement workflow

### Component 2: MIPRO Optimization Pipeline
**Core Features:**
- MIPRO (Multi-Prompt Improvement) algorithm implementation
- Automated prompt optimization using LLM feedback
- A/B testing framework for prompt comparison
- Performance metrics tracking and analysis

**Integration Points:**
- Work with existing naive baseline (v1_naive.txt)
- Generate optimized prompts (v2_optimized.txt)
- Maintain backward compatibility with existing reasoning functions

### Component 3: Evaluation System
**Metrics to Track:**
- Precision@3 for precedent selection
- Playbook quality (LLM-as-judge)
- Narrative coherence (LLM-as-judge)
- Composite score calculation (40% precision + 30% playbook + 30% narrative)

**Evaluation Framework:**
- Automated benchmark suite
- Comparison against naive baseline
- Performance regression detection
- Cost efficiency analysis (token usage optimization)

### Component 4: Production Integration
**Deployment Features:**
- Gradual rollout with feature flags
- Performance monitoring and alerting
- Rollback capability to naive baseline
- Configuration management for optimization parameters

## Success Criteria
- [ ] DSPy framework properly integrated and tested
- [ ] MIPRO optimization generating improved prompts
- [ ] Performance metrics showing improvement over naive baseline
- [ ] All existing tests continue to pass
- [ ] Production-ready with monitoring and rollback capability
- [ ] Documentation and examples for optimization workflow

## Technical Implementation Strategy
1. **Setup Phase**: Install DSPy, create optimization pipeline
2. **Baseline Phase**: Establish evaluation framework with naive prompt
3. **Optimization Phase**: Run MIPRO to generate improved prompts
4. **Validation Phase**: A/B test optimized vs baseline performance
5. **Integration Phase**: Deploy with monitoring and rollback

## Files to Create/Modify
**New Files:**
- `dealgraph/reasoning/dspy/__init__.py`
- `dealgraph/reasoning/dspy/optimizer.py` - MIPRO optimization logic
- `dealgraph/reasoning/dspy/evaluator.py` - Performance evaluation system
- `dealgraph/reasoning/dspy/config.py` - DSPy configuration management
- `tests/test_dspy_optimization.py`
- `prompts/deal_reasoner/v2_optimized.txt` - DSPy-generated optimized prompt

**Modified Files:**
- `dealgraph/reasoning/reasoner.py` - Add DSPy integration
- `tests/test_reasoning.py` - Add DSPy optimization tests

## Risk Mitigation
- **Backward Compatibility**: Maintain naive baseline as fallback
- **Performance Regression**: Implement automated testing with rollback
- **Cost Control**: Monitor token usage and optimization efficiency
- **Quality Assurance**: Multi-metric evaluation before production

## Success Metrics
- **Performance**: >10% improvement in composite score over naive baseline
- **Reliability**: 100% test pass rate with optimized prompts
- **Efficiency**: Token usage optimization >15% reduction
- **Quality**: LLM-as-judge scores >8/10 for optimized outputs

**Status**: ✅ **COMPLETED**

**Progress**:
- [x] Component 1: DSPy Framework Setup
- [x] Component 2: MIPROv2 Optimization Pipeline  
- [x] Component 3: Evaluation System
- [x] Component 4: Production Integration

**Next ExecPlan**: 3.3 (Performance Testing & Validation)
