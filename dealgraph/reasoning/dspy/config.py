# ABOUTME: DSPy configuration management for optimization settings.
# ABOUTME: Centralizes all DSPy-related configuration parameters.

import os
from typing import Optional
from dataclasses import dataclass, field
from ...config.settings import settings


@dataclass
class DSPyConfig:
    """Configuration for DSPy optimization."""
    
    # Model configuration
    model_name: str = field(default_factory=lambda: settings.DSPY_MODEL)
    api_key: str = field(default_factory=lambda: settings.CEREBRAS_API_KEY)
    base_url: str = field(default_factory=lambda: settings.CEREBRAS_BASE_URL)
    
    # Temperature settings
    optimization_temperature: float = 0.1  # Low temp for consistent optimization
    evaluation_temperature: float = 0.3    # Temp for evaluation
    
    # Token limits
    max_tokens_optimization: int = 4000
    max_tokens_evaluation: int = 2500
    
    # MIPRO parameters
    num_candidate_prompts: int = 10
    max_evaluations: int = 100
    min_improvement_threshold: float = 0.05
    
    # Performance thresholds
    min_precision_score: float = 0.6
    min_playbook_quality: float = 0.7
    min_narrative_coherence: float = 0.7
    
    # File paths
    optimization_output_dir: str = "prompts/deal_reasoner"
    benchmark_results_dir: str = "results/benchmarks"
    
    # Feature flags
    enable_verbose_logging: bool = False
    enable_performance_monitoring: bool = True
    enable_rollback_on_failure: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("Cerebras API key is required for DSPy optimization")
        if self.num_candidate_prompts < 1:
            raise ValueError("num_candidate_prompts must be >= 1")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be >= 1")
        if not (0.0 <= self.min_improvement_threshold <= 1.0):
            raise ValueError("min_improvement_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.optimization_temperature <= 2.0):
            raise ValueError("optimization_temperature must be between 0.0 and 2.0")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "optimization_temperature": self.optimization_temperature,
            "evaluation_temperature": self.evaluation_temperature,
            "max_tokens_optimization": self.max_tokens_optimization,
            "max_tokens_evaluation": self.max_tokens_evaluation,
            "num_candidate_prompts": self.num_candidate_prompts,
            "max_evaluations": self.max_evaluations,
            "min_improvement_threshold": self.min_improvement_threshold,
            "min_precision_score": self.min_precision_score,
            "min_playbook_quality": self.min_playbook_quality,
            "min_narrative_coherence": self.min_narrative_coherence
        }
    
    @classmethod
    def from_environment(cls) -> 'DSPyConfig':
        """Create configuration from environment variables."""
        return cls(
            model_name=os.getenv("DSPY_MODEL", settings.DSPY_MODEL),
            optimization_temperature=float(os.getenv("DSPY_OPTIMIZATION_TEMP", "0.1")),
            num_candidate_prompts=int(os.getenv("DSPY_CANDIDATE_PROMPTS", "10")),
            max_evaluations=int(os.getenv("DSPY_MAX_EVALUATIONS", "100")),
            min_improvement_threshold=float(os.getenv("DSPY_MIN_IMPROVEMENT", "0.05")),
            enable_verbose_logging=os.getenv("DSPY_VERBOSE", "false").lower() == "true",
            enable_performance_monitoring=os.getenv("DSPY_MONITORING", "true").lower() == "true"
        )
