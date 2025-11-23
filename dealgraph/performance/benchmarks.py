# ABOUTME: Core benchmark testing framework for evaluating prompt performance.
# ABOUTME: Provides standardized testing, statistical analysis, and result comparison.

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import json
from pathlib import Path

from dealgraph.performance.test_cases import TestCaseGenerator, DealTestCase
from dealgraph.performance.metrics import PerformanceMetrics, StatisticalAnalysis
from dealgraph.reasoning import deal_reasoner, extract_reasoning_metrics, ReasoningError
from dealgraph.data.schemas import RankedDeal, DealReasoningOutput


@dataclass
class BenchmarkResult:
    """Results from benchmark testing."""
    
    benchmark_name: str
    timestamp: datetime
    duration_seconds: float
    baseline_results: Dict[str, Any]
    optimized_results: Dict[str, Any]
    test_cases: List[DealTestCase]
    statistical_analysis: Dict[str, Any]
    improvements: Dict[str, float]
    success_rate: float
    confidence_level: float = 0.95
    total_tests: int = 0
    passed_tests: int = 0
    error_details: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "baseline_results": self.baseline_results,
            "optimized_results": self.optimized_results,
            "statistical_analysis": self.statistical_analysis,
            "improvements": self.improvements,
            "success_rate": self.success_rate,
            "confidence_level": self.confidence_level,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "error_details": self.error_details
        }


class BenchmarkSuite:
    """
    Comprehensive benchmark testing suite for prompt performance evaluation.
    
    Compares naive baseline vs. DSPy-optimized prompts using standardized test cases
    and statistical analysis.
    """
    
    def __init__(
        self,
        test_case_generator: Optional[TestCaseGenerator] = None,
        performance_metrics: Optional[PerformanceMetrics] = None,
        confidence_level: float = 0.95
    ):
        """
        Initialize benchmark suite.
        
        Args:
            test_case_generator: Generator for standardized test cases
            performance_metrics: Performance measurement utilities
            confidence_level: Statistical confidence level for tests
        """
        self.test_case_generator = test_case_generator or TestCaseGenerator()
        self.performance_metrics = performance_metrics or PerformanceMetrics()
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
        
        # Track benchmark history
        self.benchmark_history: List[BenchmarkResult] = []
    
    def run_comprehensive_benchmark(
        self,
        benchmark_name: str = "comprehensive",
        num_test_cases: int = 100,
        prompt_versions: List[str] = None,
        categories: List[str] = None
    ) -> BenchmarkResult:
        """
        Run comprehensive benchmark testing.
        
        Args:
            benchmark_name: Name for this benchmark run
            num_test_cases: Number of test cases to generate
            prompt_versions: List of prompt versions to test
            categories: Deal categories to test
            
        Returns:
            BenchmarkResult with comprehensive analysis
        """
        try:
            self.logger.info(f"Starting comprehensive benchmark: {benchmark_name}")
            start_time = time.time()
            
            # Default prompt versions
            if prompt_versions is None:
                prompt_versions = ["v1", "v2"]
            
            # Generate test cases
            test_cases = self.test_case_generator.generate_test_suite(
                num_cases=num_test_cases,
                categories=categories
            )
            
            # Run tests for each prompt version
            all_results = {}
            error_details = []
            
            for prompt_version in prompt_versions:
                self.logger.info(f"Testing prompt version: {prompt_version}")
                
                version_results = []
                success_count = 0
                version_errors = []
                
                for i, test_case in enumerate(test_cases):
                    try:
                        # Run reasoning with this prompt version
                        reasoning_output = self._run_reasoning_with_prompt(
                            query=test_case.query,
                            ranked_deals=test_case.ranked_deals,
                            prompt_version=prompt_version
                        )
                        
                        # Extract metrics
                        metrics = self.performance_metrics.evaluate_reasoning_output(
                            reasoning_output,
                            reference_data=test_case.reference_data
                        )
                        
                        version_results.append(metrics)
                        success_count += 1
                        
                        if (i + 1) % 10 == 0:
                            self.logger.debug(f"Completed {i + 1}/{len(test_cases)} tests for {prompt_version}")
                        
                    except Exception as e:
                        error_msg = f"Test case {i} failed for {prompt_version}: {str(e)}"
                        self.logger.warning(error_msg)
                        version_errors.append(error_msg)
                        error_details.append(f"{prompt_version}: {error_msg}")
                
                # Aggregate results for this version
                version_metrics = self.performance_metrics.aggregate_results(version_results)
                version_metrics["success_rate"] = success_count / len(test_cases)
                version_metrics["error_count"] = len(version_errors)
                
                all_results[prompt_version] = version_metrics
                
                self.logger.info(
                    f"Prompt {prompt_version}: {success_count}/{len(test_cases)} tests passed "
                    f"({success_count/len(test_cases)*100:.1f}%)"
                )
            
            # Calculate improvements and statistical analysis
            baseline_version = prompt_versions[0]
            optimized_versions = prompt_versions[1:]
            
            improvements = {}
            statistical_analysis = {}
            
            for optimized_version in optimized_versions:
                if baseline_version in all_results and optimized_version in all_results:
                    baseline = all_results[baseline_version]
                    optimized = all_results[optimized_version]
                    
                    # Calculate improvements
                    version_improvements = self.performance_metrics.calculate_improvements(
                        baseline, optimized
                    )
                    improvements[optimized_version] = version_improvements
                    
                    # Perform statistical analysis
                    version_stats = self._perform_statistical_analysis(
                        test_cases, baseline_version, optimized_version
                    )
                    statistical_analysis[optimized_version] = version_stats
            
            # Create benchmark result
            duration = time.time() - start_time
            total_tests = len(test_cases) * len(prompt_versions)
            passed_tests = sum(
                all_results[version].get("success_rate", 0) * len(test_cases)
                for version in prompt_versions
            )
            
            result = BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp=datetime.now(),
                duration_seconds=duration,
                baseline_results=all_results.get(baseline_version, {}),
                optimized_results={v: all_results[v] for v in optimized_versions},
                test_cases=test_cases,
                statistical_analysis=statistical_analysis,
                improvements=improvements,
                success_rate=passed_tests / total_tests if total_tests > 0 else 0,
                confidence_level=self.confidence_level,
                total_tests=total_tests,
                passed_tests=int(passed_tests),
                error_details=error_details
            )
            
            # Store in history
            self.benchmark_history.append(result)
            
            self.logger.info(
                f"Benchmark completed in {duration:.2f}s. "
                f"Success rate: {result.success_rate:.2%}, "
                f"Total tests: {result.total_tests}, "
                f"Passed: {result.passed_tests}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            raise
    
    def run_category_benchmark(
        self,
        category: str,
        num_test_cases: int = 50,
        prompt_versions: List[str] = None
    ) -> BenchmarkResult:
        """
        Run benchmark for specific deal category.
        
        Args:
            category: Deal category to benchmark
            num_test_cases: Number of test cases for this category
            prompt_versions: Prompt versions to test
            
        Returns:
            BenchmarkResult for this category
        """
        return self.run_comprehensive_benchmark(
            benchmark_name=f"category_{category}",
            num_test_cases=num_test_cases,
            prompt_versions=prompt_versions,
            categories=[category]
        )
    
    def run_cross_category_benchmark(
        self,
        categories: List[str],
        num_test_cases_per_category: int = 25,
        prompt_versions: List[str] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark across multiple categories.
        
        Args:
            categories: List of deal categories to test
            num_test_cases_per_category: Test cases per category
            prompt_versions: Prompt versions to test
            
        Returns:
            Dictionary mapping category to BenchmarkResult
        """
        results = {}
        
        for category in categories:
            self.logger.info(f"Running benchmark for category: {category}")
            
            result = self.run_category_benchmark(
                category=category,
                num_test_cases=num_test_cases_per_category,
                prompt_versions=prompt_versions
            )
            
            results[category] = result
            
            # Log category performance
            baseline_metrics = result.baseline_results
            optimized_metrics = result.optimized_results
            
            if optimized_metrics:
                optimized_version = list(optimized_metrics.keys())[0]
                improvements = result.improvements.get(optimized_version, {})
                composite_improvement = improvements.get("composite_score", 0)
                
                self.logger.info(
                    f"Category {category}: {composite_improvement:.1%} composite score improvement"
                )
        
        return results
    
    def _run_reasoning_with_prompt(
        self,
        query: str,
        ranked_deals: List[RankedDeal],
        prompt_version: str
    ) -> Dict[str, Any]:
        """Run reasoning with specific prompt version."""
        try:
            reasoning_output = deal_reasoner(
                query=query,
                ranked_deals=ranked_deals,
                prompt_version=prompt_version
            )
            
            # Convert to dictionary for metrics
            return {
                "precedents": [
                    {
                        "deal_id": p.deal_id,
                        "name": p.name,
                        "similarity_reason": p.similarity_reason
                    }
                    for p in reasoning_output.precedents
                ],
                "playbook_levers": reasoning_output.playbook_levers,
                "risk_themes": reasoning_output.risk_themes,
                "narrative_summary": reasoning_output.narrative_summary
            }
            
        except ReasoningError as e:
            raise Exception(f"Reasoning failed: {e}")
    
    def _perform_statistical_analysis(
        self,
        test_cases: List[DealTestCase],
        baseline_version: str,
        optimized_version: str
    ) -> Dict[str, Any]:
        """Perform statistical analysis comparing two prompt versions."""
        try:
            # Collect metrics for both versions
            baseline_metrics = []
            optimized_metrics = []
            
            for test_case in test_cases:
                try:
                    # Get baseline metrics
                    baseline_output = self._run_reasoning_with_prompt(
                        test_case.query, test_case.ranked_deals, baseline_version
                    )
                    baseline_metric = self.performance_metrics.evaluate_reasoning_output(
                        baseline_output, test_case.reference_data
                    )
                    baseline_metrics.append(baseline_metric["composite_score"])
                    
                    # Get optimized metrics
                    optimized_output = self._run_reasoning_with_prompt(
                        test_case.query, test_case.ranked_deals, optimized_version
                    )
                    optimized_metric = self.performance_metrics.evaluate_reasoning_output(
                        optimized_output, test_case.reference_data
                    )
                    optimized_metrics.append(optimized_metric["composite_score"])
                    
                except Exception:
                    # Skip failed tests
                    continue
            
            if len(baseline_metrics) < 10 or len(optimized_metrics) < 10:
                return {"error": "Insufficient data for statistical analysis"}
            
            # Perform statistical tests
            statistical_analysis = StatisticalAnalysis()
            
            # T-test for composite score difference
            ttest_result = statistical_analysis.paired_ttest(
                baseline_metrics, optimized_metrics
            )
            
            # Effect size calculation
            effect_size = statistical_analysis.cohens_d(baseline_metrics, optimized_metrics)
            
            # Confidence interval for improvement
            improvements = [opt - base for opt, base in zip(optimized_metrics, baseline_metrics)]
            ci_result = statistical_analysis.confidence_interval(improvements, self.confidence_level)
            
            return {
                "sample_size": len(baseline_metrics),
                "baseline_mean": statistics.mean(baseline_metrics),
                "optimized_mean": statistics.mean(optimized_metrics),
                "improvement_mean": statistics.mean(improvements),
                "ttest_p_value": ttest_result["p_value"],
                "ttest_statistic": ttest_result["statistic"],
                "cohens_d": effect_size,
                "confidence_interval": ci_result,
                "statistically_significant": ttest_result["p_value"] < (1 - self.confidence_level),
                "effect_size_interpretation": self._interpret_effect_size(effect_size)
            }
            
        except Exception as e:
            self.logger.warning(f"Statistical analysis failed: {e}")
            return {"error": str(e)}
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def save_benchmark_result(self, result: BenchmarkResult, output_path: str) -> str:
        """
        Save benchmark result to file.
        
        Args:
            result: Benchmark result to save
            output_path: Path to save the result
            
        Returns:
            Path to saved file
        """
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save result
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info(f"Benchmark result saved to: {output_path}")
        return output_path
    
    def get_benchmark_history(self) -> List[BenchmarkResult]:
        """Get history of benchmark results."""
        return self.benchmark_history.copy()
    
    def get_performance_trends(self, metric: str = "composite_score") -> Dict[str, Any]:
        """Get performance trends over time."""
        if not self.benchmark_history:
            return {"error": "No benchmark history available"}
        
        # Collect data points
        timestamps = []
        baseline_values = []
        optimized_values = []
        
        for result in self.benchmark_history:
            timestamps.append(result.timestamp)
            
            # Extract metric values
            if result.baseline_results and metric in result.baseline_results:
                baseline_values.append(result.baseline_results[metric])
            else:
                baseline_values.append(None)
            
            # Get first optimized version
            optimized_results = result.optimized_results
            if optimized_results:
                first_optimized = list(optimized_results.values())[0]
                if first_optimized and metric in first_optimized:
                    optimized_values.append(first_optimized[metric])
                else:
                    optimized_values.append(None)
            else:
                optimized_values.append(None)
        
        # Calculate trends
        trends = {
            "metric": metric,
            "time_points": [ts.isoformat() for ts in timestamps],
            "baseline_values": [v for v in baseline_values if v is not None],
            "optimized_values": [v for v in optimized_values if v is not None],
            "baseline_trend": self._calculate_trend([v for v in baseline_values if v is not None]),
            "optimized_trend": self._calculate_trend([v for v in optimized_values if v is not None])
        }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics for a list of values."""
        if len(values) < 2:
            return {"error": "Insufficient data points"}
        
        # Simple linear trend
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope using simple linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        slope = numerator / denominator if denominator != 0 else 0
        
        return {
            "slope": slope,
            "direction": "improving" if slope > 0 else "declining" if slope < 0 else "stable",
            "mean": y_mean,
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values)
        }
