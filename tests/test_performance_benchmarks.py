"""Unit tests for performance testing and validation framework."""

import pytest
import time
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import statistics

from dealgraph.performance import (
    BenchmarkSuite, BenchmarkResult,
    TestCaseGenerator, DealTestCase,
    PerformanceMetrics, StatisticalAnalysis,
    ABTestFramework, ABTestResult,
    PerformanceMonitor, MonitoringAlert, PerformanceThreshold
)
from dealgraph.performance.benchmarks import BenchmarkSuite
from dealgraph.performance.test_cases import TestCaseGenerator
from dealgraph.performance.metrics import PerformanceMetrics, StatisticalAnalysis
from dealgraph.performance.ab_testing import ABTestFramework
from dealgraph.performance.monitoring import PerformanceMonitor, PerformanceThreshold


class TestTestCaseGenerator:
    """Test test case generator functionality."""
    
    def test_initialization(self):
        """Test test case generator initialization."""
        generator = TestCaseGenerator()
        
        assert generator.data_source == 'data/raw'
        assert len(generator.query_templates) > 0
        assert len(generator.difficulty_patterns) > 0
    
    def test_get_categories(self):
        """Test getting available categories."""
        generator = TestCaseGenerator()
        categories = generator.get_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(cat, str) for cat in categories)
    
    def test_get_difficulty_levels(self):
        """Test getting available difficulty levels."""
        generator = TestCaseGenerator()
        difficulties = generator.get_difficulty_levels()
        
        assert isinstance(difficulties, list)
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties
    
    def test_generate_test_suite(self):
        """Test generating test suite."""
        generator = TestCaseGenerator()
        
        # Test with small number of cases
        test_cases = generator.generate_test_suite(num_cases=5)
        
        assert len(test_cases) == 5
        assert all(isinstance(tc, DealTestCase) for tc in test_cases)
        assert all(tc.case_id for tc in test_cases)
        assert all(tc.query for tc in test_cases)
    
    def test_generate_edge_cases(self):
        """Test generating edge cases."""
        generator = TestCaseGenerator()
        edge_cases = generator.generate_edge_cases()
        
        assert len(edge_cases) > 0
        assert all(isinstance(ec, DealTestCase) for ec in edge_cases)
        
        # Check for specific edge cases
        edge_case_ids = [ec.case_id for ec in edge_cases]
        assert any("no_candidates" in case_id for case_id in edge_case_ids)
        assert any("single_candidate" in case_id for case_id in edge_case_ids)
    
    def test_validate_test_case(self):
        """Test test case validation."""
        generator = TestCaseGenerator()
        
        # Valid test case
        valid_case = DealTestCase(
            case_id="test_001",
            query="Test query",
            ranked_deals=[],
            category="technology",
            difficulty="easy"
        )
        
        validation = generator.validate_test_case(valid_case)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        
        # Invalid test case
        invalid_case = DealTestCase(
            case_id="",
            query="",
            ranked_deals=[],
            category="invalid_category",
            difficulty="invalid_difficulty"
        )
        
        validation = generator.validate_test_case(invalid_case)
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0


class TestPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_evaluate_reasoning_output(self):
        """Test evaluating reasoning output."""
        metrics = PerformanceMetrics()
        
        reasoning_output = {
            "precedents": [
                {"deal_id": "test1", "name": "Test Deal", "similarity_reason": "Strong sector alignment"}
            ],
            "playbook_levers": ["Operational improvements", "Market expansion"],
            "risk_themes": ["Competition risk"],
            "narrative_summary": "Strong precedent analysis with strategic value creation opportunities"
        }
        
        result = metrics.evaluate_reasoning_output(reasoning_output)
        
        assert "precision_at_3" in result
        assert "playbook_quality" in result
        assert "narrative_coherence" in result
        assert "composite_score" in result
        
        # Check scores are in valid range
        for score in result.values():
            if isinstance(score, (int, float)):
                assert 0.0 <= score <= 1.0
    
    def test_aggregate_results(self):
        """Test aggregating multiple results."""
        metrics = PerformanceMetrics()
        
        results = [
            {"composite_score": 0.7, "precision_at_3": 0.6},
            {"composite_score": 0.8, "precision_at_3": 0.7},
            {"composite_score": 0.6, "precision_at_3": 0.5}
        ]
        
        aggregated = metrics.aggregate_results(results)
        
        assert aggregated["composite_score"] == 0.7
        assert "composite_score_mean" in aggregated
        assert "composite_score_std" in aggregated
    
    def test_calculate_improvements(self):
        """Test calculating improvements between versions."""
        metrics = PerformanceMetrics()
        
        baseline = {"composite_score": 0.6, "precision_at_3": 0.5}
        optimized = {"composite_score": 0.7, "precision_at_3": 0.6}
        
        improvements = metrics.calculate_improvements(baseline, optimized)
        
        assert "composite_score_improvement" in improvements
        assert "composite_score_absolute" in improvements
        assert improvements["composite_score_improvement"] > 0
    
    def test_heuristic_scoring(self):
        """Test heuristic scoring methods."""
        metrics = PerformanceMetrics()
        
        # Test playbook quality scoring
        good_levers = [
            "Increase operational efficiency through process optimization",
            "Expand market presence via strategic acquisitions",
            "Improve cost structure and margins"
        ]
        
        score = metrics._heuristic_playbook_score(good_levers)
        assert score > 0.5
        
        # Test narrative quality scoring
        good_narrative = "Strong precedents demonstrate strategic value creation opportunities through operational optimization and market expansion strategies."
        
        score = metrics._heuristic_narrative_score(good_narrative)
        assert score > 0.5


class TestStatisticalAnalysis:
    """Test statistical analysis functionality."""
    
    def test_paired_ttest(self):
        """Test paired t-test."""
        stats_analysis = StatisticalAnalysis()
        
        baseline = [0.6, 0.7, 0.5, 0.8, 0.6]
        optimized = [0.7, 0.8, 0.6, 0.9, 0.7]
        
        result = stats_analysis.paired_ttest(baseline, optimized)
        
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "mean_difference" in result
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        stats_analysis = StatisticalAnalysis()
        
        values = [0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.9, 0.5]
        
        ci = stats_analysis.confidence_interval(values)
        
        assert ci is not None
        assert len(ci) == 2
        assert ci[0] < ci[1]
    
    def test_cohens_d(self):
        """Test Cohen's d effect size calculation."""
        stats_analysis = StatisticalAnalysis()
        
        baseline = [0.6, 0.7, 0.5, 0.8, 0.6]
        optimized = [0.7, 0.8, 0.6, 0.9, 0.7]
        
        effect_size = stats_analysis.cohens_d(baseline, optimized)
        
        assert isinstance(effect_size, float)
        assert effect_size != 0  # Should show some effect


class TestBenchmarkSuite:
    """Test benchmark suite functionality."""
    
    def test_initialization(self):
        """Test benchmark suite initialization."""
        generator = TestCaseGenerator()
        metrics = PerformanceMetrics()
        
        suite = BenchmarkSuite(
            test_case_generator=generator,
            performance_metrics=metrics
        )
        
        assert suite.test_case_generator == generator
        assert suite.performance_metrics == metrics
        assert suite.confidence_level == 0.95
    
    @patch('dealgraph.performance.benchmarks.deal_reasoner')
    def test_run_comprehensive_benchmark_mock(self, mock_deal_reasoner):
        """Test running comprehensive benchmark with mocked reasoning."""
        # Mock reasoning output
        mock_deal_reasoner.return_value = Mock(
            precedents=[Mock(deal_id="test", name="Test", similarity_reason="Test")],
            playbook_levers=["lever"],
            risk_themes=["risk"],
            narrative_summary="Test summary"
        )
        
        generator = TestCaseGenerator()
        metrics = PerformanceMetrics()
        suite = BenchmarkSuite(
            test_case_generator=generator,
            performance_metrics=metrics
        )
        
        # Run benchmark with very small test case
        result = suite.run_comprehensive_benchmark(
            benchmark_name="test",
            num_test_cases=2,
            prompt_versions=["v1"]
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "test"
        assert result.success_rate > 0
    
    def test_save_benchmark_result(self):
        """Test saving benchmark result."""
        generator = TestCaseGenerator()
        suite = BenchmarkSuite(test_case_generator=generator)
        
        # Create mock result
        result = BenchmarkResult(
            benchmark_name="test",
            timestamp=datetime.now(),
            duration_seconds=10.0,
            baseline_results={"composite_score": 0.7},
            optimized_results={},
            test_cases=[],
            statistical_analysis={},
            improvements={},
            success_rate=0.9
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_result.json"
            
            saved_path = suite.save_benchmark_result(result, str(output_path))
            
            assert saved_path == str(output_path)
            assert output_path.exists()
            
            # Verify saved content
            with open(output_path) as f:
                saved_data = json.load(f)
            
            assert saved_data["benchmark_name"] == "test"
            assert saved_data["duration_seconds"] == 10.0
    
    def test_get_performance_trends(self):
        """Test getting performance trends."""
        generator = TestCaseGenerator()
        suite = BenchmarkSuite(test_case_generator=generator)
        
        # Create mock benchmark history
        mock_result = BenchmarkResult(
            benchmark_name="test",
            timestamp=datetime.now(),
            duration_seconds=10.0,
            baseline_results={"composite_score": 0.7},
            optimized_results={},
            test_cases=[],
            statistical_analysis={},
            improvements={},
            success_rate=0.9
        )
        
        suite.benchmark_history = [mock_result]
        
        trends = suite.get_performance_trends("composite_score")
        
        assert "metric" in trends
        assert trends["metric"] == "composite_score"


class TestABTestFramework:
    """Test A/B testing framework."""
    
    def test_initialization(self):
        """Test A/B framework initialization."""
        metrics = PerformanceMetrics()
        stats_analysis = StatisticalAnalysis()
        
        framework = ABTestFramework(
            performance_metrics=metrics,
            statistical_analysis=stats_analysis
        )
        
        assert framework.performance_metrics == metrics
        assert framework.statistical_analysis == stats_analysis
        assert len(framework.active_tests) == 0
    
    def test_start_ab_test(self):
        """Test starting A/B test."""
        framework = ABTestFramework()
        
        variant_a = {"name": "baseline", "prompt_version": "v1"}
        variant_b = {"name": "optimized", "prompt_version": "v2"}
        
        test_id = framework.start_ab_test(
            test_id="test_001",
            variant_a_config=variant_a,
            variant_b_config=variant_b,
            min_sample_size=10
        )
        
        assert test_id == "test_001"
        assert test_id in framework.active_tests
        
        test_result = framework.active_tests[test_id]
        assert test_result.variant_a_name == "baseline"
        assert test_result.variant_b_name == "optimized"
    
    def test_assign_variant(self):
        """Test variant assignment."""
        framework = ABTestFramework()
        
        framework.start_ab_test(
            test_id="test_002",
            variant_a_config={"name": "control"},
            variant_b_config={"name": "treatment"}
        )
        
        # Test assignment
        variant = framework.assign_variant("test_002", "user123")
        
        assert variant in ["variant_a", "variant_b"]
    
    def test_record_result(self):
        """Test recording A/B test results."""
        framework = ABTestFramework()
        
        framework.start_ab_test(
            test_id="test_003",
            variant_a_config={"name": "control"},
            variant_b_config={"name": "treatment"}
        )
        
        # Mock reasoning to avoid actual API calls
        with patch('dealgraph.performance.ab_testing.deal_reasoner') as mock_deal_reasoner:
            mock_deal_reasoner.return_value = Mock(
                precedents=[],
                playbook_levers=[],
                risk_themes=[],
                narrative_summary="Test"
            )
            
            success = framework.record_result(
                test_id="test_003",
                variant="variant_a",
                query="test query",
                ranked_deals=[]
            )
            
            assert success is True
            assert len(framework.active_tests["test_003"].variant_a_results) > 0
    
    def test_stop_ab_test(self):
        """Test stopping A/B test."""
        framework = ABTestFramework()
        
        framework.start_ab_test(
            test_id="test_004",
            variant_a_config={"name": "control"},
            variant_b_config={"name": "treatment"}
        )
        
        success = framework.stop_ab_test("test_004")
        
        assert success is True
        assert "test_004" not in framework.active_tests
        assert len(framework.completed_tests) == 1


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        
        assert monitor.monitoring_active is False
        assert len(monitor.performance_history) > 0
        assert len(monitor.alert_thresholds) > 0
        assert monitor.total_requests == 0
    
    def test_record_request_success(self):
        """Test recording successful request."""
        monitor = PerformanceMonitor()
        
        with patch('dealgraph.performance.monitoring.deal_reasoner') as mock_deal_reasoner:
            mock_deal_reasoner.return_value = Mock(
                precedents=[],
                playbook_levers=[],
                risk_themes=[],
                narrative_summary="Test summary"
            )
            
            metrics = monitor.record_request(
                query="test query",
                ranked_deals=[],
                response_time_seconds=1.5
            )
            
            assert monitor.total_requests == 1
            assert monitor.successful_requests == 1
            assert "composite_score" in metrics
    
    def test_record_request_failure(self):
        """Test recording failed request."""
        monitor = PerformanceMonitor()
        
        with patch('dealgraph.performance.monitoring.deal_reasoner') as mock_deal_reasoner:
            mock_deal_reasoner.side_effect = Exception("API Error")
            
            metrics = monitor.record_request(
                query="test query",
                ranked_deals=[]
            )
            
            assert monitor.total_requests == 1
            assert monitor.failed_requests == 1
            assert "composite_score" in metrics
    
    def test_custom_threshold(self):
        """Test adding custom threshold."""
        monitor = PerformanceMonitor()
        
        original_count = len(monitor.alert_thresholds)
        
        custom_threshold = PerformanceThreshold(
            metric_name="custom_metric",
            warning_threshold=0.8,
            critical_threshold=0.6
        )
        
        monitor.add_custom_threshold(custom_threshold)
        
        assert len(monitor.alert_thresholds) == original_count + 1
        assert any(t.metric_name == "custom_metric" for t in monitor.alert_thresholds)
    
    def test_get_current_metrics(self):
        """Test getting current metrics."""
        monitor = PerformanceMonitor()
        
        # Add some mock data
        monitor.performance_history["composite_score"].extend([0.7, 0.8, 0.6])
        monitor.total_requests = 3
        monitor.successful_requests = 2
        
        metrics = monitor.get_current_metrics()
        
        assert "composite_score" in metrics
        assert "summary" in metrics
        assert metrics["summary"]["total_requests"] == 3
        assert metrics["summary"]["successful_requests"] == 2
    
    def test_alert_callback(self):
        """Test alert callback functionality."""
        alerts_received = []
        
        def alert_callback(alert: MonitoringAlert):
            alerts_received.append(alert)
        
        monitor = PerformanceMonitor(alert_callback=alert_callback)
        
        # Create a test alert
        test_alert = MonitoringAlert(
            alert_id="test_001",
            timestamp=datetime.now(),
            alert_type="test",
            severity="medium",
            message="Test alert",
            metric_name="test_metric",
            current_value=0.5,
            threshold_value=0.7
        )
        
        monitor.active_alerts["test_001"] = test_alert
        
        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0]["message"] == "Test alert"


class TestIntegration:
    """Integration tests for complete performance testing workflow."""
    
    def test_complete_benchmark_workflow(self):
        """Test complete benchmark testing workflow."""
        generator = TestCaseGenerator()
        metrics = PerformanceMetrics()
        stats_analysis = StatisticalAnalysis()
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            test_case_generator=generator,
            performance_metrics=metrics
        )
        
        # Generate small test suite
        test_cases = generator.generate_test_suite(num_cases=3)
        
        assert len(test_cases) == 3
        assert all(isinstance(tc, DealTestCase) for tc in test_cases)
        
        # Test metrics calculation
        sample_output = {
            "precedents": [{"deal_id": "test", "name": "Test", "similarity_reason": "Test"}],
            "playbook_levers": ["lever"],
            "risk_themes": ["risk"],
            "narrative_summary": "Test summary"
        }
        
        sample_metrics = metrics.evaluate_reasoning_output(sample_output)
        
        assert "composite_score" in sample_metrics
        assert 0.0 <= sample_metrics["composite_score"] <= 1.0
    
    def test_complete_ab_test_workflow(self):
        """Test complete A/B testing workflow."""
        metrics = PerformanceMetrics()
        framework = ABTestFramework(performance_metrics=metrics)
        
        # Start test
        test_id = framework.start_ab_test(
            test_id="integration_test",
            variant_a_config={"name": "control", "prompt_version": "v1"},
            variant_b_config={"name": "treatment", "prompt_version": "v2"},
            min_sample_size=5
        )
        
        assert test_id == "integration_test"
        assert framework.get_test_status(test_id) is not None
        
        # Stop test
        success = framework.stop_ab_test(test_id)
        assert success is True
        
        final_status = framework.get_test_status(test_id)
        assert final_status["status"] == "completed"
    
    def test_monitoring_integration(self):
        """Test monitoring system integration."""
        monitor = PerformanceMonitor()
        
        # Test metrics collection
        mock_metrics = {
            "composite_score": 0.7,
            "precision_at_3": 0.6,
            "playbook_quality": 0.8,
            "narrative_coherence": 0.7
        }
        
        monitor._record_performance_metrics(mock_metrics, response_time_seconds=2.0)
        
        # Verify data was recorded
        assert len(monitor.performance_history["composite_score"]) > 0
        assert len(monitor.performance_history["response_time"]) > 0
        
        # Test getting current status
        current_metrics = monitor.get_current_metrics()
        assert "composite_score" in current_metrics
        assert "response_time" in current_metrics
        assert current_metrics["composite_score"]["current"] == 0.7
