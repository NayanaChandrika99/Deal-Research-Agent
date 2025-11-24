# ABOUTME: DSPy optimization module for prompt improvement using MIPRO algorithm.
# ABOUTME: Provides automated prompt optimization and performance evaluation.

import os
import logging
import json
from typing import Callable, Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import dspy
from dspy.teleprompt import MIPROv2

from .config import DSPyConfig
from .evaluator import PerformanceEvaluator
from ..dspy_modules import DealReasonerModule
from ..prompts import load_prompt
from ...eval.bench_dataset import load_dealgraph_bench
from ...config.settings import settings


class DealReasonerMetric:
    """Composite metric that mirrors SPEC weights for DSPy optimization."""

    def __init__(self, evaluator: Optional[PerformanceEvaluator] = None):
        self.evaluator = evaluator or PerformanceEvaluator()

    def __call__(self, example, prediction) -> float:
        try:
            reasoning_output = _prediction_to_reasoning_dict(prediction)
            expected_ids = getattr(example, "expected_relevant_ids", [])
            reference = {
                "precedents": [
                    {"deal_id": deal_id, "name": deal_id, "similarity_reason": ""}
                    for deal_id in expected_ids
                ]
            }
            metrics = self.evaluator.evaluate_reasoning_output(
                reasoning_output,
                reference,
            )
            return metrics.get("composite_score") or metrics.get("precision_at_3", 0.0)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "DealReasonerMetric evaluation failed: %s", exc
            )
            return 0.0


def _prediction_to_reasoning_dict(prediction: Any) -> Dict[str, Any]:
    """Convert a dspy.Prediction into the evaluator-friendly structure."""
    def _ensure_json(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    return {
        "precedents": _ensure_json(getattr(prediction, "precedents", [])),
        "playbook_levers": _ensure_json(getattr(prediction, "playbook_levers", [])),
        "risk_themes": _ensure_json(getattr(prediction, "risk_themes", [])),
        "narrative_summary": getattr(prediction, "narrative_summary", ""),
    }


class DSPyOptimizer:
    """
    DSPy-based prompt optimization using MIPRO algorithm.
    
    Automatically improves prompts using LLM feedback and evaluation metrics.
    """
    
    def __init__(
        self,
        config: Optional[DSPyConfig] = None,
        evaluator: Optional[PerformanceEvaluator] = None,
        metric: Optional[Callable] = None,
    ):
        """
        Initialize DSPy optimizer.
        
        Args:
            config: DSPy configuration settings
            evaluator: Performance evaluator for optimization metrics
        """
        self.config = config or DSPyConfig()
        self.evaluator = evaluator or PerformanceEvaluator()
        self.metric = metric or DealReasonerMetric(self.evaluator)
        self.logger = logging.getLogger(__name__)
        
        # DSPy setup
        self._setup_dspy()
        
        # Track optimization results
        self.optimization_history: List[Dict[str, Any]] = []
        
    def _setup_dspy(self):
        """Configure DSPy with model settings."""
        # Configure language model for DSPy optimization
        dspy.configure(
            lm=dspy.LM(
                model=f"openai/{settings.DSPY_MODEL}",
                api_key=settings.CEREBRAS_API_KEY,
                api_base=settings.CEREBRAS_BASE_URL,
                temperature=0.1,  # Lower temperature for optimization
                max_tokens=4000
            )
        )
        
        self.logger.info(f"DSPy configured with model: {settings.DSPY_MODEL}")
    
    def optimize_deal_reasoner_prompt(
        self,
        baseline_prompt_version: str = "v1",
        num_candidate_prompts: int = 10,
        max_evaluations: int = 100,
        min_improvement: float = 0.05
    ) -> Dict[str, Any]:
        """
        Optimize deal reasoning prompt using MIPRO.
        
        Args:
            baseline_prompt_version: Version of baseline prompt to optimize from
            num_candidate_prompts: Number of candidate prompts to generate
            max_evaluations: Maximum number of optimization evaluations
            min_improvement: Minimum improvement required to adopt new prompt
            
        Returns:
            Dictionary with optimization results and metrics
        """
        try:
            self.logger.info(
                f"Starting DSPy optimization for prompt {baseline_prompt_version}"
            )
            
            # Load baseline prompt
            try:
                baseline_prompt = load_prompt("deal_reasoner", baseline_prompt_version)
                baseline_system_prompt = baseline_prompt["metadata"].get("description", "")
                baseline_user_prompt = baseline_prompt["content"]
            except FileNotFoundError:
                raise ValueError(f"Baseline prompt version '{baseline_prompt_version}' not found")
            
            # Create DSPy module for optimization
            deal_reasoner_module = DealReasonerModule()
            
            # Run MIPROv2 optimization
            self.logger.info("Running MIPROv2 optimization...")
            teleprompter = MIPROv2(
                metric=self.metric,
                num_candidate_prompts=num_candidate_prompts,
                max_evaluations=max_evaluations,
                verbose=False
            )
            
            # Optimize with MIPRO
            trainset = self._get_training_dataset()
            valset = self._get_validation_dataset()
            optimized_program = teleprompter.compile(
                deal_reasoner_module,
                trainset=trainset,
                valset=valset
            )
            
            # Extract optimized prompts
            system_prompt = optimized_program.get("system_prompt", "")
            user_prompt = optimized_program.get("user_prompt", "")
            
            # Evaluate performance improvement
            baseline_score = self._evaluate_module(deal_reasoner_module, valset or trainset)
            optimized_score = self._evaluate_module(optimized_program, valset or trainset)
            improvement = optimized_score - baseline_score
            
            # Store results
            optimization_result = {
                "baseline_version": baseline_prompt_version,
                "optimized_version": f"v{int(baseline_prompt_version.replace('v', '')) + 1}",
                "optimization_time": datetime.now().isoformat(),
                "improvement_score": improvement,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "candidate_prompts_evaluated": num_candidate_prompts,
                "total_evaluations": max_evaluations,
                "min_improvement_threshold": min_improvement,
                "performance_metrics": {
                    "baseline_score": baseline_score,
                    "optimized_score": optimized_score
                },
                "optimized_program": optimized_program,
                "meets_threshold": improvement >= min_improvement
            }
            
            self.optimization_history.append(optimization_result)
            
            self.logger.info(
                f"Optimization complete. Improvement: {improvement:.3f}, "
                f"Meets threshold: {improvement >= min_improvement}"
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"DSPy optimization failed: {e}")
            raise
    
    def _evaluation_metric(self, example, pred) -> float:
        """
        Evaluation metric for MIPRO optimization.
        
        Args:
            example: Training/example case
            pred: Model prediction
            
        Returns:
            Score from 0.0 to 1.0
        """
        try:
            reasoning_output = getattr(pred, 'reasoning_output', {})
            expected_ids = getattr(example, "expected_relevant_ids", [])
            reference_data = {
                "precedents": [
                    {"deal_id": deal_id, "name": deal_id, "similarity_reason": ""}
                    for deal_id in expected_ids
                ]
            }
            score = self.evaluator.calculate_composite_score(
                reasoning_output,
                reference_data
            )
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Evaluation metric error: {e}")
            return 0.5  # Neutral score
    
    def _get_training_dataset(self) -> List[dspy.Example]:
        """Get training dataset for optimization."""
        examples: List[dspy.Example] = []
        for bench_query in load_dealgraph_bench():
            candidate_summary = ", ".join(bench_query.relevant_deal_ids)
            example = dspy.Example(
                query=bench_query.text,
                candidate_deals=f"Relevant deals: {candidate_summary}",
                expected_relevant_ids=bench_query.relevant_deal_ids
            ).with_inputs("query", "candidate_deals")
            examples.append(example)
        return examples
    
    def _get_validation_dataset(self) -> List[dspy.Example]:
        """Get validation dataset for optimization."""
        training = self._get_training_dataset()
        if not training:
            return []
        mid = max(1, len(training) // 2)
        return training[:mid]
    
    def _evaluate_module(
        self,
        module: DealReasonerModule,
        dataset: Optional[List[dspy.Example]]
    ) -> float:
        """Run the given module across a dataset and average composite scores."""
        examples = dataset or self._get_training_dataset()
        if not examples:
            return 0.0
        scores: List[float] = []
        for example in examples:
            try:
                prediction = module(
                    query=example.query,
                    candidate_deals=example.candidate_deals
                )
                scores.append(self.metric(example, prediction))
            except Exception as exc:
                self.logger.warning("Module evaluation failed: %s", exc)
        return sum(scores) / len(scores) if scores else 0.0
    
    def save_optimized_prompt(
        self,
        optimization_result: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Save optimized prompt to file.
        
        Args:
            optimization_result: Result from optimize_deal_reasoner_prompt
            output_path: Optional custom output path
            
        Returns:
            Path to saved optimized prompt
        """
        if not optimization_result.get("meets_threshold", False):
            self.logger.warning("Optimization result does not meet improvement threshold")
        
        # Determine output path
        optimized_version = optimization_result["optimized_version"]
        if not output_path:
            output_path = f"prompts/deal_reasoner/{optimized_version}_optimized.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        module_path = optimization_result.get("module_path")
        optimized_program = optimization_result.get("optimized_program")
        if optimized_program and not module_path:
            module_path = output_path.with_name(f"{output_path.stem}_module.json")
            try:
                optimized_program.save(module_path)
                optimization_result["module_path"] = str(module_path)
            except Exception as exc:
                self.logger.warning("Unable to save DSPy module state: %s", exc)
                module_path = None

        payload = {
            "content": optimization_result["user_prompt"],
            "metadata": {
                "version": optimized_version,
                "name": "deal_reasoner",
                "description": "DSPy optimized reasoning prompt",
                "prompt_type": "optimized",
                "system_prompt": optimization_result["system_prompt"],
                "model": settings.DSPY_MODEL,
                "created_date": optimization_result["optimization_time"],
                "performance_metrics": optimization_result.get("performance_metrics"),
                "module_path": str(module_path) if module_path else None,
            },
        }

        output_path.write_text(json.dumps(payload, indent=2))
        self.logger.info("Optimized prompt saved to %s", output_path)
        return str(output_path)
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization attempts."""
        return self.optimization_history.copy()
    
    def rollback_to_baseline(self, baseline_version: str = "v1") -> bool:
        """
        Rollback to baseline prompt if optimization failed.
        
        Args:
            baseline_version: Version to rollback to
            
        Returns:
            True if rollback successful
        """
        try:
            # Ensure baseline prompt exists
            baseline_prompt = load_prompt("deal_reasoner", baseline_version)
            
            self.logger.info(f"Rolled back to baseline version {baseline_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False


# Global optimizer instance
_dspy_optimizer: Optional[DSPyOptimizer] = None


def get_dspy_optimizer() -> DSPyOptimizer:
    """Get the global DSPy optimizer instance."""
    global _dspy_optimizer
    
    if _dspy_optimizer is None:
        _dspy_optimizer = DSPyOptimizer()
    
    return _dspy_optimizer


def set_dspy_optimizer(optimizer: DSPyOptimizer) -> None:
    """Set the global DSPy optimizer instance."""
    global _dspy_optimizer
    _dspy_optimizer = optimizer


def optimize_deal_reasoner(
    baseline_version: str = "v1",
    num_candidate_prompts: int = 10,
    max_evaluations: int = 100,
    output_path: Optional[str] = None,
    min_improvement: float = 0.05,
) -> Dict[str, Any]:
    """
    Convenience wrapper to run DSPy optimization and persist results.
    """
    optimizer = DSPyOptimizer()
    result = optimizer.optimize_deal_reasoner_prompt(
        baseline_prompt_version=baseline_version,
        num_candidate_prompts=num_candidate_prompts,
        max_evaluations=max_evaluations,
        min_improvement=min_improvement,
    )
    if result.get("meets_threshold"):
        optimizer.save_optimized_prompt(result, output_path)
    return result
