# ABOUTME: A/B testing framework for real-time prompt performance comparison.
# ABOUTME: Provides statistical analysis, automated winner selection, and monitoring.

import logging
import time
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import statistics

from dealgraph.performance.metrics import PerformanceMetrics, StatisticalAnalysis
from dealgraph.reasoning import deal_reasoner, ReasoningError


@dataclass
class ABTestResult:
    """Result from A/B test."""
    
    test_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, stopped, error
    variant_a_name: str = "control"
    variant_b_name: str = "treatment"
    variant_a_results: List[Dict[str, float]] = field(default_factory=list)
    variant_b_results: List[Dict[str, float]] = field(default_factory=list)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    winner: Optional[str] = None
    confidence_level: float = 0.95
    min_sample_size: int = 30
    current_sample_size: int = 0
    error_details: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Get test duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Get overall success rate."""
        total_attempts = len(self.variant_a_results) + len(self.variant_b_results)
        if total_attempts == 0:
            return 0.0
        
        successful_attempts = sum(
            1 for results in self.variant_a_results + self.variant_b_results
            if results.get("composite_score", 0) > 0
        )
        
        return successful_attempts / total_attempts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_id": self.test_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "variant_a_name": self.variant_a_name,
            "variant_b_name": self.variant_b_name,
            "statistical_analysis": self.statistical_analysis,
            "winner": self.winner,
            "confidence_level": self.confidence_level,
            "min_sample_size": self.min_sample_size,
            "current_sample_size": self.current_sample_size,
            "duration_seconds": self.duration_seconds,
            "success_rate": self.success_rate,
            "error_details": self.error_details,
            "variant_a_results_count": len(self.variant_a_results),
            "variant_b_results_count": len(self.variant_b_results)
        }


class ABTestFramework:
    """
    A/B testing framework for comparing prompt versions in real-time.
    
    Provides statistical analysis, automated winner selection, and performance monitoring.
    """
    
    def __init__(
        self,
        performance_metrics: Optional[PerformanceMetrics] = None,
        statistical_analysis: Optional[StatisticalAnalysis] = None,
        confidence_level: float = 0.95
    ):
        """
        Initialize A/B testing framework.
        
        Args:
            performance_metrics: Performance measurement utilities
            statistical_analysis: Statistical analysis utilities
            confidence_level: Statistical confidence level
        """
        self.performance_metrics = performance_metrics or PerformanceMetrics()
        self.statistical_analysis = statistical_analysis or StatisticalAnalysis()
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
        
        # Track active and completed tests
        self.active_tests: Dict[str, ABTestResult] = {}
        self.completed_tests: List[ABTestResult] = []
    
    def start_ab_test(
        self,
        test_id: str,
        variant_a_config: Dict[str, Any],
        variant_b_config: Dict[str, Any],
        min_sample_size: int = 30,
        confidence_level: float = None,
        auto_stop: bool = True
    ) -> str:
        """
        Start a new A/B test.
        
        Args:
            test_id: Unique identifier for the test
            variant_a_config: Configuration for variant A (control)
            variant_b_config: Configuration for variant B (treatment)
            min_sample_size: Minimum sample size per variant
            confidence_level: Override default confidence level
            auto_stop: Whether to automatically stop when statistically significant
            
        Returns:
            Test ID
        """
        if test_id in self.active_tests:
            raise ValueError(f"Test {test_id} is already active")
        
        confidence_level = confidence_level or self.confidence_level
        
        test_result = ABTestResult(
            test_id=test_id,
            start_time=datetime.now(),
            variant_a_name=variant_a_config.get("name", "control"),
            variant_b_name=variant_b_config.get("name", "treatment"),
            confidence_level=confidence_level,
            min_sample_size=min_sample_size
        )
        
        # Store configurations
        test_result.variant_a_config = variant_a_config
        test_result.variant_b_config = variant_b_config
        test_result.auto_stop = auto_stop
        
        self.active_tests[test_id] = test_result
        
        self.logger.info(
            f"Started A/B test {test_id}: {test_result.variant_a_name} vs {test_result.variant_b_name}"
        )
        
        return test_id
    
    def assign_variant(self, test_id: str, user_id: str = None) -> str:
        """
        Assign a variant to a user/request.
        
        Args:
            test_id: ID of the active test
            user_id: Optional user identifier for consistent assignment
            
        Returns:
            Variant name ("variant_a" or "variant_b")
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} is not active")
        
        test_result = self.active_tests[test_id]
        
        # Use user_id for consistent assignment if provided
        if user_id:
            random.seed(hash(user_id + test_id))
        
        # 50/50 split
        variant = "variant_a" if random.random() < 0.5 else "variant_b"
        
        return variant
    
    def record_result(
        self,
        test_id: str,
        variant: str,
        query: str,
        ranked_deals,
        metrics: Optional[Dict[str, float]] = None,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a result for an A/B test.
        
        Args:
            test_id: ID of the active test
            variant: Which variant was used ("variant_a" or "variant_b")
            query: User query that was processed
            ranked_deals: Ranked deals used
            metrics: Optional pre-calculated metrics
            reference_data: Optional reference data
            
        Returns:
            True if result was recorded successfully
        """
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} is not active")
            
            test_result = self.active_tests[test_id]
            
            # Calculate metrics if not provided
            if metrics is None:
                # Run reasoning with appropriate variant
                variant_config = getattr(test_result, f"{variant}_config")
                prompt_version = variant_config.get("prompt_version", "v1")
                
                try:
                    reasoning_output = deal_reasoner(
                        query=query,
                        ranked_deals=ranked_deals,
                        prompt_version=prompt_version
                    )
                    
                    # Convert to metrics
                    metrics = self.performance_metrics.evaluate_reasoning_output(
                        reasoning_output, reference_data
                    )
                    
                except ReasoningError as e:
                    self.logger.warning(f"Reasoning failed for {test_id}: {e}")
                    metrics = self.performance_metrics._get_default_metrics()
            
            # Store result
            result_list = getattr(test_result, f"{variant}_results")
            result_list.append(metrics)
            test_result.current_sample_size += 1
            
            # Check if we should stop
            if test_result.auto_stop and self._should_stop_test(test_result):
                self.stop_ab_test(test_id)
            
            self.logger.debug(f"Recorded {variant} result for {test_id}, sample size: {test_result.current_sample_size}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record A/B test result: {e}")
            test_result.error_details.append(f"Result recording error: {str(e)}")
            return False
    
    def _should_stop_test(self, test_result: ABTestResult) -> bool:
        """
        Determine if a test should be stopped based on statistical significance.
        
        Args:
            test_result: Test result to evaluate
            
        Returns:
            True if test should be stopped
        """
        # Check minimum sample size
        if len(test_result.variant_a_results) < test_result.min_sample_size:
            return False
        if len(test_result.variant_b_results) < test_result.min_sample_size:
            return False
        
        # Check if enough data points for statistical test
        if len(test_result.variant_a_results) < 10 or len(test_result.variant_b_results) < 10:
            return False
        
        # Extract composite scores
        variant_a_scores = [r.get("composite_score", 0) for r in test_result.variant_a_results]
        variant_b_scores = [r.get("composite_score", 0) for r in test_result.variant_b_results]
        
        # Perform statistical test
        ttest_result = self.statistical_analysis.paired_ttest(
            variant_a_scores, variant_b_scores, test_result.confidence_level
        )
        
        # Update test result
        test_result.statistical_analysis = {
            "last_update": datetime.now().isoformat(),
            "sample_size_a": len(variant_a_scores),
            "sample_size_b": len(variant_b_scores),
            "mean_a": statistics.mean(variant_a_scores),
            "mean_b": statistics.mean(variant_b_scores),
            "ttest_result": ttest_result,
            "effect_size": self.statistical_analysis.cohens_d(variant_a_scores, variant_b_scores)
        }
        
        # Determine if we have a winner
        if ttest_result["significant"]:
            mean_a = ttest_result["mean_difference"] + statistics.mean(variant_a_scores)
            mean_b = statistics.mean(variant_b_scores)
            
            if mean_b > mean_a:
                test_result.winner = test_result.variant_b_name
            else:
                test_result.winner = test_result.variant_a_name
            
            return True
        
        return False
    
    def stop_ab_test(self, test_id: str, force_winner: str = None) -> bool:
        """
        Stop an A/B test and analyze final results.
        
        Args:
            test_id: ID of the test to stop
            force_winner: Optionally force a winner ("variant_a" or "variant_b")
            
        Returns:
            True if test was stopped successfully
        """
        try:
            if test_id not in self.active_tests:
                return False
            
            test_result = self.active_tests[test_id]
            test_result.end_time = datetime.now()
            test_result.status = "completed"
            
            # Force winner if specified
            if force_winner:
                if force_winner == "variant_a":
                    test_result.winner = test_result.variant_a_name
                elif force_winner == "variant_b":
                    test_result.winner = test_result.variant_b_name
            
            # Perform final analysis
            self._perform_final_analysis(test_result)
            
            # Move to completed tests
            self.completed_tests.append(test_result)
            del self.active_tests[test_id]
            
            self.logger.info(
                f"Stopped A/B test {test_id}, winner: {test_result.winner}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop A/B test: {e}")
            return False
    
    def _perform_final_analysis(self, test_result: ABTestResult):
        """Perform final statistical analysis on test results."""
        try:
            # Extract metrics for analysis
            variant_a_scores = [r.get("composite_score", 0) for r in test_result.variant_a_results]
            variant_b_scores = [r.get("composite_score", 0) for r in test_result.variant_b_results]
            
            if not variant_a_scores or not variant_b_scores:
                return
            
            # Comprehensive statistical analysis
            ttest_result = self.statistical_analysis.paired_ttest(
                variant_a_scores, variant_b_scores, test_result.confidence_level
            )
            
            effect_size = self.statistical_analysis.cohens_d(variant_a_scores, variant_b_scores)
            
            # Calculate confidence intervals
            ci_a = self.statistical_analysis.confidence_interval(variant_a_scores, test_result.confidence_level)
            ci_b = self.statistical_analysis.confidence_interval(variant_b_scores, test_result.confidence_level)
            
            # Calculate improvements
            mean_a = statistics.mean(variant_a_scores)
            mean_b = statistics.mean(variant_b_scores)
            improvement = (mean_b - mean_a) / mean_a if mean_a != 0 else 0
            
            test_result.statistical_analysis = {
                "test_completed": True,
                "sample_size_a": len(variant_a_scores),
                "sample_size_b": len(variant_b_scores),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "std_a": statistics.stdev(variant_a_scores) if len(variant_a_scores) > 1 else 0,
                "std_b": statistics.stdev(variant_b_scores) if len(variant_b_scores) > 1 else 0,
                "confidence_interval_a": ci_a,
                "confidence_interval_b": ci_b,
                "ttest_result": ttest_result,
                "effect_size": effect_size,
                "absolute_improvement": mean_b - mean_a,
                "relative_improvement": improvement,
                "interpretation": self._interpret_results(
                    ttest_result, effect_size, improvement, test_result.confidence_level
                )
            }
            
        except Exception as e:
            self.logger.warning(f"Final analysis failed: {e}")
            test_result.statistical_analysis = {"error": str(e)}
    
    def _interpret_results(
        self,
        ttest_result: Dict[str, Any],
        effect_size: float,
        improvement: float,
        confidence_level: float
    ) -> str:
        """Interpret test results."""
        if not ttest_result.get("significant", False):
            return f"No statistically significant difference at {confidence_level:.0%} confidence level"
        
        # Interpret effect size
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            effect_interpretation = "negligible"
        elif abs_effect < 0.5:
            effect_interpretation = "small"
        elif abs_effect < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # Determine direction
        if improvement > 0:
            direction = "improvement"
        else:
            direction = "degradation"
        
        return (
            f"Statistically significant {direction} with {effect_interpretation} effect size "
            f"({improvement:.1%} relative change, p-value: {ttest_result.get('p_value', 0):.4f})"
        )
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an A/B test."""
        if test_id in self.active_tests:
            test_result = self.active_tests[test_id]
            return test_result.to_dict()
        elif test_id in [t.test_id for t in self.completed_tests]:
            test_result = next(t for t in self.completed_tests if t.test_id == test_id)
            return test_result.to_dict()
        else:
            return None
    
    def get_all_active_tests(self) -> List[Dict[str, Any]]:
        """Get status of all active tests."""
        return [test.to_dict() for test in self.active_tests.values()]
    
    def get_all_completed_tests(self) -> List[Dict[str, Any]]:
        """Get status of all completed tests."""
        return [test.to_dict() for test in self.completed_tests]
    
    def save_test_result(self, test_result: ABTestResult, output_path: str) -> str:
        """Save A/B test result to file."""
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save result
        with open(output_path, 'w') as f:
            json.dump(test_result.to_dict(), f, indent=2)
        
        self.logger.info(f"A/B test result saved to: {output_path}")
        return output_path
    
    def cleanup_old_tests(self, max_age_days: int = 30) -> int:
        """
        Clean up old completed tests.
        
        Args:
            max_age_days: Maximum age in days to keep tests
            
        Returns:
            Number of tests cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        original_count = len(self.completed_tests)
        self.completed_tests = [
            test for test in self.completed_tests
            if test.end_time and test.end_time > cutoff_date
        ]
        
        cleaned_count = original_count - len(self.completed_tests)
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old A/B test results")
        
        return cleaned_count
