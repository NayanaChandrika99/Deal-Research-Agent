"""Reasoning layer for DealGraph - LLM-based deal analysis and insight generation."""

from .prompts import PromptRegistry, load_prompt, DEAL_REASONER_NAIVE_PROMPT
from .llm_client import LLMClient, LLMClientError, get_llm_client
from .reasoner import (
    deal_reasoner,
    analyze_deals_with_naive_prompt,
    extract_reasoning_metrics,
    ReasoningError,
    dspy_optimize_prompt,
    dspy_evaluate_performance,
    compare_reasoning_prompts,
    dspy_get_optimization_history,
    dspy_rollback_to_baseline
)

__all__ = [
    "PromptRegistry",
    "load_prompt",
    "DEAL_REASONER_NAIVE_PROMPT",
    "LLMClient",
    "LLMClientError",
    "get_llm_client",
    "deal_reasoner",
    "analyze_deals_with_naive_prompt",
    "extract_reasoning_metrics",
    "dspy_optimize_prompt",
    "dspy_evaluate_performance",
    "compare_reasoning_prompts",
    "dspy_get_optimization_history",
    "dspy_rollback_to_baseline",
    "ReasoningError"
]
