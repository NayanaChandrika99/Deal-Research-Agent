# ABOUTME: Performance evaluation system for DSPy-optimized prompts.
# ABOUTME: Provides comprehensive metrics for evaluating reasoning quality.

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..llm_client import get_llm_client, LLMClientError
from ...data.schemas import DealReasoningOutput, Precedent
from .config import DSPyConfig


class PerformanceEvaluator:
    """
    Evaluates performance of optimized prompts using multiple metrics.
    
    Supports precision@3, playbook quality, narrative coherence, and composite scoring.
    """
    
    def __init__(self, config: Optional[DSPyConfig] = None):
        """
        Initialize performance evaluator.
        
        Args:
            config: DSPy configuration settings
        """
        self.config = config or DSPyConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM client for evaluation
        self.llm_client = get_llm_client()
        
        # Track evaluation results
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_reasoning_output(
        self,
        reasoning_output: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate reasoning output using multiple metrics.
        
        Args:
            reasoning_output: Output from deal reasoning
            reference_data: Optional reference data for ground truth
            
        Returns:
            Dictionary with metric scores (0.0 to 1.0)
        """
        try:
            metrics = {}
            
            # Calculate precision@3 for precedents
            metrics["precision_at_3"] = self._calculate_precision_at_3(
                reasoning_output, reference_data
            )
            
            # Evaluate playbook lever quality
            metrics["playbook_quality"] = self._evaluate_playbook_quality(
                reasoning_output, reference_data
            )
            
            # Evaluate narrative coherence
            metrics["narrative_coherence"] = self._evaluate_narrative_coherence(
                reasoning_output, reference_data
            )
            
            # Calculate composite score
            metrics["composite_score"] = self.calculate_composite_score(
                reasoning_output, reference_data, metrics
            )
            
            # Store evaluation result
            evaluation_result = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "reasoning_output": reasoning_output,
                "reference_data": reference_data
            }
            
            self.evaluation_history.append(evaluation_result)
            
            self.logger.debug(f"Evaluation complete: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            return self._get_default_metrics()
    
    def calculate_composite_score(
        self,
        reasoning_output: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate composite score using weighted metrics.
        
        Args:
            reasoning_output: Model output to evaluate
            reference_data: Optional reference for ground truth
            metrics: Optional pre-calculated metrics
            
        Returns:
            Composite score from 0.0 to 1.0
        """
        if metrics is None:
            metrics = self.evaluate_reasoning_output(reasoning_output, reference_data)
        
        # Weights from specification: 40% precision + 30% playbook + 30% narrative
        weights = {
            "precision_at_3": 0.4,
            "playbook_quality": 0.3,
            "narrative_coherence": 0.3
        }
        
        composite_score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                composite_score += metrics[metric] * weight
        
        return min(max(composite_score, 0.0), 1.0)  # Clamp between 0.0 and 1.0
    
    def _calculate_precision_at_3(
        self,
        reasoning_output: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate precision@3 for precedent selection.
        
        Args:
            reasoning_output: Model output with precedents
            reference_data: Optional ground truth precedents
            
        Returns:
            Precision score from 0.0 to 1.0
        """
        try:
            precedents = reasoning_output.get("precedents", [])
            if not precedents:
                return 0.0
            
            # If no reference data, use heuristic scoring
            if not reference_data:
                return self._calculate_precision_heuristic(precedents)
            
            # Calculate precision using reference data
            reference_precedents = reference_data.get("precedents", [])
            if not reference_precedents:
                return self._calculate_precision_heuristic(precedents)
            
            # Count matches in top 3 precedents
            reference_ids = {p.get("deal_id") for p in reference_precedents}
            top_3_precedents = precedents[:3]
            matches = sum(1 for p in top_3_precedents if p.get("deal_id") in reference_ids)
            
            return min(matches / 3.0, 1.0)  # Normalize to 0.0-1.0
            
        except Exception as e:
            self.logger.warning(f"Precision@3 calculation failed: {e}")
            return self._calculate_precision_heuristic(reasoning_output.get("precedents", []))
    
    def _calculate_precision_heuristic(self, precedents: List[Dict[str, Any]]) -> float:
        """Calculate precision using heuristic rules."""
        if not precedents:
            return 0.0
        
        score = 0.0
        
        # Check precedent quality
        for i, precedent in enumerate(precedents[:3]):
            # Boost score for detailed explanations
            similarity_reason = precedent.get("similarity_reason", "")
            if len(similarity_reason) > 50:
                score += 0.2
            
            # Boost score for meaningful names
            name = precedent.get("name", "")
            if name and len(name) > 3:
                score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_playbook_quality(
        self,
        reasoning_output: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate playbook lever quality using LLM-as-judge.
        
        Args:
            reasoning_output: Model output with playbook levers
            reference_data: Optional reference playbook levers
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        try:
            playbook_levers = reasoning_output.get("playbook_levers", [])
            if not playbook_levers:
                return 0.0
            
            # If LLM evaluation available, use it
            quality_score = self._llm_evaluate_playbook_quality(playbook_levers)
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Playbook quality evaluation failed: {e}")
            return self._heuristic_playbook_score(reasoning_output.get("playbook_levers", []))
    
    def _evaluate_narrative_coherence(
        self,
        reasoning_output: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate narrative coherence using LLM-as-judge.
        
        Args:
            reasoning_output: Model output with narrative summary
            reference_data: Optional reference narrative
            
        Returns:
            Coherence score from 0.0 to 1.0
        """
        try:
            narrative = reasoning_output.get("narrative_summary", "")
            if not narrative:
                return 0.0
            
            # If LLM evaluation available, use it
            coherence_score = self._llm_evaluate_narrative_coherence(narrative)
            
            return coherence_score
            
        except Exception as e:
            self.logger.warning(f"Narrative coherence evaluation failed: {e}")
            return self._heuristic_narrative_score(narrative)
    
    def _llm_evaluate_playbook_quality(self, playbook_levers: List[str]) -> float:
        """Use LLM to evaluate playbook lever quality."""
        try:
            evaluation_prompt = f"""
            Evaluate the quality of these playbook levers on a scale from 0 to 10:
            
            Playbook Levers:
            {chr(10).join(f"- {lever}" for lever in playbook_levers)}
            
            Consider:
            1. Specificity and actionability
            2. Relevance to private equity value creation
            3. Diversity of strategies
            4. Implementation feasibility
            
            Return only a number from 0 to 10.
            """
            
            response = self.llm_client.complete_text(
                system_prompt="You are an expert evaluating private equity playbook levers.",
                user_prompt=evaluation_prompt,
                temperature=0.1,
                max_tokens=100
            )
            
            # Parse response to get score
            score_text = response.strip()
            score = float(score_text) / 10.0  # Convert 0-10 to 0-1
            
            return max(0.0, min(score, 1.0))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"LLM playbook evaluation failed: {e}")
            return 0.5  # Neutral score
    
    def _llm_evaluate_narrative_coherence(self, narrative: str) -> float:
        """Use LLM to evaluate narrative coherence."""
        try:
            evaluation_prompt = f"""
            Evaluate the coherence and quality of this narrative summary on a scale from 0 to 10:
            
            Narrative: {narrative}
            
            Consider:
            1. Logical flow and structure
            2. Executive-level insights
            3. Actionable conclusions
            4. Professional writing quality
            
            Return only a number from 0 to 10.
            """
            
            response = self.llm_client.complete_text(
                system_prompt="You are an expert evaluating executive narrative summaries.",
                user_prompt=evaluation_prompt,
                temperature=0.1,
                max_tokens=100
            )
            
            # Parse response to get score
            score_text = response.strip()
            score = float(score_text) / 10.0  # Convert 0-10 to 0-1
            
            return max(0.0, min(score, 1.0))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"LLM narrative evaluation failed: {e}")
            return 0.5  # Neutral score
    
    def _heuristic_playbook_score(self, playbook_levers: List[str]) -> float:
        """Calculate playbook quality using heuristic rules."""
        if not playbook_levers:
            return 0.0
        
        score = 0.0
        
        for lever in playbook_levers:
            lever = lever.strip()
            if not lever:
                continue
            
            # Boost for action-oriented language
            action_words = ["improve", "optimize", "reduce", "increase", "streamline", "expand"]
            if any(word in lever.lower() for word in action_words):
                score += 0.2
            
            # Boost for strategic specificity
            if len(lever) > 20:
                score += 0.1
            
            # Cap individual lever contribution
            score = min(score, 0.5)
        
        return min(score, 1.0)
    
    def _heuristic_narrative_score(self, narrative: str) -> float:
        """Calculate narrative coherence using heuristic rules."""
        if not narrative or len(narrative) < 50:
            return 0.0
        
        score = 0.0
        
        # Boost for reasonable length
        if 100 <= len(narrative) <= 500:
            score += 0.3
        elif len(narrative) > 500:
            score += 0.2
        
        # Boost for executive language
        executive_words = ["strategic", "investment", "value", "opportunity", "market"]
        if any(word in narrative.lower() for word in executive_words):
            score += 0.2
        
        # Boost for structured language
        if "\n" in narrative or "." in narrative:
            score += 0.2
        
        # Boost for specific insights
        if "precedent" in narrative.lower() or "playbook" in narrative.lower():
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when evaluation fails."""
        return {
            "precision_at_3": 0.5,
            "playbook_quality": 0.5,
            "narrative_coherence": 0.5,
            "composite_score": 0.5
        }
    
    def compare_prompts(
        self,
        prompt_a: Dict[str, Any],
        prompt_b: Dict[str, Any],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compare two prompts on test cases.
        
        Args:
            prompt_a: First prompt data
            prompt_b: Second prompt data
            test_cases: List of test cases to evaluate on
            
        Returns:
            Dictionary with comparison results
        """
        results = {
            "prompt_a_scores": [],
            "prompt_b_scores": [],
            "improvement_scores": [],
            "average_improvement": 0.0
        }
        
        for test_case in test_cases:
            # Evaluate prompt A
            score_a = self.evaluate_reasoning_output(
                prompt_a.get("reasoning_output", {}),
                test_case
            )["composite_score"]
            
            # Evaluate prompt B
            score_b = self.evaluate_reasoning_output(
                prompt_b.get("reasoning_output", {}),
                test_case
            )["composite_score"]
            
            # Calculate improvement
            improvement = score_b - score_a
            
            # Store results
            results["prompt_a_scores"].append(score_a)
            results["prompt_b_scores"].append(score_b)
            results["improvement_scores"].append(improvement)
        
        # Calculate average improvement
        results["average_improvement"] = sum(results["improvement_scores"]) / len(results["improvement_scores"])
        
        return results
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of evaluation results."""
        return self.evaluation_history.copy()
    
    def save_evaluation_report(self, output_path: str) -> str:
        """
        Save evaluation report to file.
        
        Args:
            output_path: Path to save evaluation report
            
        Returns:
            Path to saved report
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "evaluation_count": len(self.evaluation_history),
            "evaluations": self.evaluation_history,
            "summary": self._generate_evaluation_summary()
        }
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to: {output_path}")
        return output_path
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from evaluation history."""
        if not self.evaluation_history:
            return {"error": "No evaluations recorded"}
        
        metrics_scores = {
            "precision_at_3": [],
            "playbook_quality": [],
            "narrative_coherence": [],
            "composite_score": []
        }
        
        for evaluation in self.evaluation_history:
            for metric in metrics_scores.keys():
                if metric in evaluation["metrics"]:
                    metrics_scores[metric].append(evaluation["metrics"][metric])
        
        summary = {}
        for metric, scores in metrics_scores.items():
            if scores:
                summary[metric] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        return summary
