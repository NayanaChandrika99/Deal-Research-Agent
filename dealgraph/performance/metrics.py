# ABOUTME: Performance metrics calculation and statistical analysis utilities.
# ABOUTME: Provides comprehensive measurement of prompt performance and significance testing.

import logging
import statistics
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import numpy as np


@dataclass
class StatisticalResult:
    """Result from statistical analysis."""
    statistic: float
    p_value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    significant: bool = False
    interpretation: str = ""


class PerformanceMetrics:
    """
    Performance metrics calculator for prompt evaluation.
    
    Calculates comprehensive metrics including precision, quality scores,
    and composite scores with statistical analysis.
    """
    
    def __init__(self):
        """Initialize performance metrics calculator."""
        self.logger = logging.getLogger(__name__)
        
        # Metric weights for composite scoring
        self.metric_weights = {
            "precision_at_3": 0.4,
            "playbook_quality": 0.3,
            "narrative_coherence": 0.3
        }
    
    def evaluate_reasoning_output(
        self,
        reasoning_output: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate reasoning output performance.
        
        Args:
            reasoning_output: Output from deal reasoning
            reference_data: Optional reference data for ground truth
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            metrics = {}
            
            # Calculate individual metrics
            metrics["precision_at_3"] = self._calculate_precision_at_3(
                reasoning_output, reference_data
            )
            metrics["playbook_quality"] = self._calculate_playbook_quality(
                reasoning_output, reference_data
            )
            metrics["narrative_coherence"] = self._calculate_narrative_coherence(
                reasoning_output, reference_data
            )
            
            # Calculate composite score
            metrics["composite_score"] = self._calculate_composite_score(metrics)
            
            # Additional metrics
            metrics["precedent_count"] = len(reasoning_output.get("precedents", []))
            metrics["playbook_lever_count"] = len(reasoning_output.get("playbook_levers", []))
            metrics["risk_theme_count"] = len(reasoning_output.get("risk_themes", []))
            metrics["narrative_length"] = len(reasoning_output.get("narrative_summary", ""))
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Performance evaluation failed: {e}")
            return self._get_default_metrics()
    
    def _calculate_precision_at_3(
        self,
        reasoning_output: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate precision@3 for precedent selection."""
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
            
            # Check for structured information
            if i < len(precedents):
                precedent_text = str(precedents[i]).lower()
                if any(word in precedent_text for word in ["strategic", "market", "operational"]):
                    score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_playbook_quality(
        self,
        reasoning_output: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate playbook lever quality score."""
        try:
            playbook_levers = reasoning_output.get("playbook_levers", [])
            if not playbook_levers:
                return 0.0
            
            # Use heuristic scoring (LLM evaluation would be preferred but more complex)
            return self._heuristic_playbook_score(playbook_levers)
            
        except Exception as e:
            self.logger.warning(f"Playbook quality calculation failed: {e}")
            return self._heuristic_playbook_score(reasoning_output.get("playbook_levers", []))
    
    def _calculate_narrative_coherence(
        self,
        reasoning_output: Dict[str, Any],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate narrative coherence score."""
        try:
            narrative = reasoning_output.get("narrative_summary", "")
            if not narrative:
                return 0.0
            
            # Use heuristic scoring
            return self._heuristic_narrative_score(narrative)
            
        except Exception as e:
            self.logger.warning(f"Narrative coherence calculation failed: {e}")
            return self._heuristic_narrative_score(reasoning_output.get("narrative_summary", ""))
    
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
            action_words = [
                "improve", "optimize", "reduce", "increase", "streamline", "expand",
                "implement", "enhance", "develop", "leverage", "accelerate"
            ]
            if any(word in lever.lower() for word in action_words):
                score += 0.3
            
            # Boost for strategic specificity
            if len(lever) > 30:
                score += 0.2
            
            # Boost for specific metrics or numbers
            if any(char.isdigit() for char in lever):
                score += 0.1
            
            # Boost for professional terminology
            professional_words = [
                "revenue", "margin", "ebitda", "cash flow", "synergy", "scale",
                "efficiency", "productivity", "market share", "competitive"
            ]
            if any(word in lever.lower() for word in professional_words):
                score += 0.2
            
            # Cap individual lever contribution
            score = min(score, 0.8)
        
        return min(score, 1.0)
    
    def _heuristic_narrative_score(self, narrative: str) -> float:
        """Calculate narrative coherence using heuristic rules."""
        if not narrative or len(narrative) < 50:
            return 0.0
        
        score = 0.0
        
        # Boost for reasonable length
        if 150 <= len(narrative) <= 800:
            score += 0.3
        elif len(narrative) > 800:
            score += 0.2
        elif len(narrative) > 100:
            score += 0.1
        
        # Boost for executive language
        executive_words = [
            "strategic", "investment", "value", "opportunity", "market", "competitive",
            "leverage", "optimize", "maximize", "sustainable", "growth"
        ]
        if any(word in narrative.lower() for word in executive_words):
            score += 0.3
        
        # Boost for structured language
        if "." in narrative:
            score += 0.2
        
        if "\n" in narrative or ";" in narrative:
            score += 0.1
        
        # Boost for specific insights
        insight_words = ["precedent", "playbook", "strategy", "analysis", "recommend"]
        if any(word in narrative.lower() for word in insight_words):
            score += 0.2
        
        # Boost for forward-looking language
        future_words = ["future", "potential", "opportunity", "growth", "expansion"]
        if any(word in narrative.lower() for word in future_words):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score using weighted metrics."""
        composite_score = 0.0
        
        for metric, weight in self.metric_weights.items():
            if metric in metrics:
                composite_score += metrics[metric] * weight
        
        return min(max(composite_score, 0.0), 1.0)  # Clamp between 0.0 and 1.0
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when evaluation fails."""
        return {
            "precision_at_3": 0.5,
            "playbook_quality": 0.5,
            "narrative_coherence": 0.5,
            "composite_score": 0.5,
            "precedent_count": 0,
            "playbook_lever_count": 0,
            "risk_theme_count": 0,
            "narrative_length": 0
        }
    
    def aggregate_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate multiple evaluation results."""
        if not results:
            return self._get_default_metrics()
        
        aggregated = {}
        
        # Calculate aggregate statistics for each metric
        all_metrics = set()
        for result in results:
            all_metrics.update(result.keys())
        
        for metric in all_metrics:
            values = [result.get(metric, 0) for result in results if metric in result]
            
            if values:
                aggregated[f"{metric}_mean"] = statistics.mean(values)
                aggregated[f"{metric}_median"] = statistics.median(values)
                aggregated[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0
                aggregated[f"{metric}_min"] = min(values)
                aggregated[f"{metric}_max"] = max(values)
                
                # Use mean as the primary aggregated value
                aggregated[metric] = aggregated[f"{metric}_mean"]
        
        return aggregated
    
    def calculate_improvements(
        self,
        baseline: Dict[str, float],
        optimized: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate improvements between baseline and optimized results."""
        improvements = {}
        
        # Calculate improvements for common metrics
        common_metrics = set(baseline.keys()) & set(optimized.keys())
        
        for metric in common_metrics:
            if isinstance(baseline[metric], (int, float)) and isinstance(optimized[metric], (int, float)):
                baseline_value = baseline[metric]
                optimized_value = optimized[metric]
                
                if baseline_value != 0:
                    relative_improvement = (optimized_value - baseline_value) / abs(baseline_value)
                    improvements[f"{metric}_improvement"] = relative_improvement
                    improvements[f"{metric}_absolute"] = optimized_value - baseline_value
                else:
                    improvements[f"{metric}_improvement"] = 0.0
                    improvements[f"{metric}_absolute"] = 0.0
        
        return improvements


class StatisticalAnalysis:
    """
    Statistical analysis utilities for performance testing.
    
    Provides statistical tests, effect size calculations, and confidence intervals.
    """
    
    def __init__(self):
        """Initialize statistical analysis utilities."""
        self.logger = logging.getLogger(__name__)
    
    def paired_ttest(
        self,
        baseline_values: List[float],
        optimized_values: List[float],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform paired t-test comparing baseline vs. optimized.
        
        Args:
            baseline_values: List of baseline metric values
            optimized_values: List of optimized metric values
            confidence_level: Confidence level for the test
            
        Returns:
            Dictionary with test results
        """
        try:
            if len(baseline_values) != len(optimized_values):
                raise ValueError("Baseline and optimized value lists must have same length")
            
            if len(baseline_values) < 2:
                raise ValueError("Need at least 2 paired values for t-test")
            
            # Calculate differences
            differences = [opt - base for opt, base in zip(optimized_values, baseline_values)]
            
            # Perform paired t-test
            statistic, p_value = stats.ttest_rel(optimized_values, baseline_values)
            
            # Determine significance
            alpha = 1 - confidence_level
            significant = p_value < alpha
            
            return {
                "statistic": statistic,
                "p_value": p_value,
                "significant": significant,
                "alpha": alpha,
                "confidence_level": confidence_level,
                "mean_difference": statistics.mean(differences),
                "std_difference": statistics.stdev(differences) if len(differences) > 1 else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Paired t-test failed: {e}")
            return {
                "error": str(e),
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False
            }
    
    def confidence_interval(
        self,
        values: List[float],
        confidence_level: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate confidence interval for a list of values.
        
        Args:
            values: List of values
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound) or None if calculation fails
        """
        try:
            if len(values) < 2:
                return None
            
            # Use scipy.stats for confidence interval
            mean = statistics.mean(values)
            sem = stats.sem(values)  # Standard error of the mean
            
            # Get t-critical value
            degrees_freedom = len(values) - 1
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
            
            # Calculate margin of error
            margin_of_error = t_critical * sem
            
            # Calculate confidence interval
            lower_bound = mean - margin_of_error
            upper_bound = mean + margin_of_error
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
            return None
    
    def cohens_d(
        self,
        baseline_values: List[float],
        optimized_values: List[float]
    ) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            baseline_values: Baseline values
            optimized_values: Optimized values
            
        Returns:
            Cohen's d effect size
        """
        try:
            if len(baseline_values) < 2 or len(optimized_values) < 2:
                return 0.0
            
            # Calculate means
            baseline_mean = statistics.mean(baseline_values)
            optimized_mean = statistics.mean(optimized_values)
            
            # Calculate pooled standard deviation
            baseline_std = statistics.stdev(baseline_values)
            optimized_std = statistics.stdev(optimized_values)
            
            n1, n2 = len(baseline_values), len(optimized_values)
            pooled_std = math.sqrt(((n1 - 1) * baseline_std**2 + (n2 - 1) * optimized_std**2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            # Calculate Cohen's d
            cohens_d = (optimized_mean - baseline_mean) / pooled_std
            
            return cohens_d
            
        except Exception as e:
            self.logger.warning(f"Cohen's d calculation failed: {e}")
            return 0.0
    
    def power_analysis(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> Dict[str, Any]:
        """
        Perform power analysis to determine required sample size.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level
            power: Desired power (1 - beta)
            
        Returns:
            Dictionary with power analysis results
        """
        try:
            # Use scipy.stats for power analysis
            from scipy.stats import norm
            
            # Calculate critical values
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            
            # Calculate required sample size per group
            n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            
            return {
                "effect_size": effect_size,
                "alpha": alpha,
                "power": power,
                "required_sample_size_per_group": math.ceil(n_per_group),
                "total_required_sample_size": math.ceil(2 * n_per_group)
            }
            
        except Exception as e:
            self.logger.warning(f"Power analysis failed: {e}")
            return {
                "error": str(e),
                "effect_size": effect_size,
                "alpha": alpha,
                "power": power
            }
    
    def bootstrap_confidence_interval(
        self,
        values: List[float],
        statistic_func,
        num_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Optional[Dict[str, float]]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            values: Original values
            statistic_func: Function to calculate statistic
            num_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Dictionary with bootstrap results
        """
        try:
            if len(values) < 2:
                return None
            
            # Calculate original statistic
            original_stat = statistic_func(values)
            
            # Generate bootstrap samples
            bootstrap_stats = []
            for _ in range(num_bootstrap):
                # Resample with replacement
                bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(bootstrap_stat)
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)
            
            return {
                "original_statistic": original_stat,
                "bootstrap_mean": statistics.mean(bootstrap_stats),
                "bootstrap_std": statistics.stdev(bootstrap_stats),
                "confidence_interval": (ci_lower, ci_upper),
                "num_bootstrap_samples": num_bootstrap
            }
            
        except Exception as e:
            self.logger.warning(f"Bootstrap confidence interval failed: {e}")
            return None
