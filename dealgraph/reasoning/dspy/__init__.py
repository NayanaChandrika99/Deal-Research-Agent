# ABOUTME: DSPy optimization system for prompt improvement and evaluation.
# ABOUTME: Provides MIPRO-based optimization with performance tracking.

from .optimizer import DSPyOptimizer, get_dspy_optimizer, set_dspy_optimizer
from .config import DSPyConfig
from .evaluator import PerformanceEvaluator

__all__ = [
    "DSPyOptimizer",
    "get_dspy_optimizer",
    "set_dspy_optimizer",
    "DSPyConfig",
    "PerformanceEvaluator"
]
