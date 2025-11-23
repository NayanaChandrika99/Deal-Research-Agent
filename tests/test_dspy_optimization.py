"""Unit tests for DSPy optimization functionality."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from dealgraph.reasoning.dspy import (
    DSPyOptimizer,
    DSPyConfig,
    PerformanceEvaluator,
    get_dspy_optimizer,
    set_dspy_optimizer
)
from dealgraph.reasoning.dspy.config import DSPyConfig
from dealgraph.reasoning.reasoner import (
    dspy_optimize_prompt,
    dspy_evaluate_performance,
    compare_reasoning_prompts,
    dspy_get_optimization_history,
    dspy_rollback_to_baseline,
    ReasoningError
)


class TestDSPyConfig:
    """Test DSPy configuration functionality."""
    
    def test_config_initialization(self):
        """Test configuration initialization with defaults."""
        config = DSPyConfig()
        
        assert config.model_name == "llama3.1-8b"
        assert config.optimization_temperature == 0.1
        assert config.num_candidate_prompts == 10
        assert config.max_evaluations == 100
        assert config.min_improvement_threshold == 0.05
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = DSPyConfig(
            optimization_temperature=0.5,
            num_candidate_prompts=5,
            min_improvement_threshold=0.1
        )
        assert config.optimization_temperature == 0.5
        assert config.num_candidate_prompts == 5
        assert config.min_improvement_threshold == 0.1
    
    def test_config_validation_errors(self):
        """Test configuration validation errors."""
        # Invalid temperature
        with pytest.raises(ValueError):
            DSPyConfig(optimization_temperature=3.0)
        
        # Invalid candidate prompts
        with pytest.raises(ValueError):
            DSPyConfig(num_candidate_prompts=0)
        
        # Invalid improvement threshold
        with pytest.raises(ValueError):
            DSPyConfig(min_improvement_threshold=1.5)
    
    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = DSPyConfig(
            model_name="test-model",
            optimization_temperature=0.2
        )
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test-model"
        assert config_dict["optimization_temperature"] == 0.2
        assert "num_candidate_prompts" in config_dict
    
    @patch.dict('os.environ', {
        'DSPY_OPTIMIZATION_TEMP': '0.3',
        'DSPY_CANDIDATE_PROMPTS': '15',
        'DSPY_MIN_IMPROVEMENT': '0.1',
        'DSPY_VERBOSE': 'true'
    })
    def test_config_from_environment(self):
        """Test configuration from environment variables."""
        config = DSPyConfig.from_environment()
        
        assert config.optimization_temperature == 0.3
        assert config.num_candidate_prompts == 15
        assert config.min_improvement_threshold == 0.1
        assert config.enable_verbose_logging is True


class TestPerformanceEvaluator:
    """Test performance evaluator functionality."""
    
    def test_evaluate_reasoning_output(self):
        """Test evaluation of reasoning output."""
        evaluator = PerformanceEvaluator()
        
        reasoning_output = {
            "precedents": [
                {"deal_id": "test1", "name": "Test Deal 1", "similarity_reason": "Strong sector match"},
                {"deal_id": "test2", "name": "Test Deal 2", "similarity_reason": "Good precedent"}
            ],
            "playbook_levers": ["Operational improvements", "Strategic acquisitions"],
            "risk_themes": ["Market competition", "Integration risk"],
            "narrative_summary": "Strong precedents with strategic value creation opportunities"
        }
        
        metrics = evaluator.evaluate_reasoning_output(reasoning_output)
        
        assert "precision_at_3" in metrics
        assert "playbook_quality" in metrics
        assert "narrative_coherence" in metrics
        assert "composite_score" in metrics
        
        # Check scores are in valid range
        for score in metrics.values():
            assert 0.0 <= score <= 1.0
    
    def test_calculate_composite_score(self):
        """Test composite score calculation."""
        evaluator = PerformanceEvaluator()
        
        reasoning_output = {
            "precedents": [{"deal_id": "test1", "name": "Test", "similarity_reason": "Test"}],
            "playbook_levers": ["lever1"],
            "risk_themes": ["risk1"],
            "narrative_summary": "Test summary"
        }
        
        metrics = {
            "precision_at_3": 0.8,
            "playbook_quality": 0.7,
            "narrative_coherence": 0.6
        }
        
        composite_score = evaluator.calculate_composite_score(
            reasoning_output, None, metrics
        )
        
        # Should be 40% * 0.8 + 30% * 0.7 + 30% * 0.6 = 0.32 + 0.21 + 0.18 = 0.71
        expected_score = 0.4 * 0.8 + 0.3 * 0.7 + 0.3 * 0.6
        assert abs(composite_score - expected_score) < 0.01
    
    def test_evaluate_empty_output(self):
        """Test evaluation of empty or missing output."""
        evaluator = PerformanceEvaluator()
        
        empty_output = {
            "precedents": [],
            "playbook_levers": [],
            "risk_themes": [],
            "narrative_summary": ""
        }
        
        metrics = evaluator.evaluate_reasoning_output(empty_output)
        
        # Should return default scores for empty output
        assert all(0.0 <= score <= 1.0 for score in metrics.values())
    
    def test_heuristic_scoring(self):
        """Test heuristic scoring methods."""
        evaluator = PerformanceEvaluator()
        
        # Test heuristic playbook scoring
        playbook_levers = [
            "Increase operational efficiency through process optimization",
            "Expand market presence via strategic acquisitions",
            "Improve cost structure"
        ]
        
        score = evaluator._heuristic_playbook_score(playbook_levers)
        assert 0.0 <= score <= 1.0
        
        # Test heuristic narrative scoring
        narrative = "Strong precedents show strategic value creation opportunities through operational improvements and market expansion. The analysis indicates robust investment potential."
        
        score = evaluator._heuristic_narrative_score(narrative)
        assert 0.0 <= score <= 1.0
    
    def test_evaluation_history(self):
        """Test evaluation history tracking."""
        evaluator = PerformanceEvaluator()
        
        reasoning_output = {
            "precedents": [{"deal_id": "test", "name": "Test", "similarity_reason": "Test"}],
            "playbook_levers": ["lever"],
            "risk_themes": ["risk"],
            "narrative_summary": "Test"
        }
        
        # Run multiple evaluations
        evaluator.evaluate_reasoning_output(reasoning_output)
        evaluator.evaluate_reasoning_output(reasoning_output)
        
        history = evaluator.get_evaluation_history()
        assert len(history) == 2
        assert all("metrics" in entry for entry in history)
    
    def test_save_evaluation_report(self):
        """Test saving evaluation report."""
        evaluator = PerformanceEvaluator()
        
        reasoning_output = {
            "precedents": [{"deal_id": "test", "name": "Test", "similarity_reason": "Test"}],
            "playbook_levers": ["lever"],
            "risk_themes": ["risk"],
            "narrative_summary": "Test"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "evaluation_report.json"
            
            # Run evaluation and save report
            evaluator.evaluate_reasoning_output(reasoning_output)
            saved_path = evaluator.save_evaluation_report(str(report_path))
            
            assert saved_path == str(report_path)
            assert report_path.exists()
            
            # Verify report content
            with open(report_path) as f:
                report_data = json.load(f)
            
            assert "timestamp" in report_data
            assert "evaluation_count" in report_data
            assert "evaluations" in report_data
            assert "summary" in report_data


class TestDSPyOptimizer:
    """Test DSPy optimizer functionality."""
    
    @patch('dealgraph.reasoning.dspy.optimizer.dspy')
    @patch('dealgraph.reasoning.dspy.optimizer.settings')
    def test_optimizer_initialization(self, mock_settings, mock_dspy):
        """Test optimizer initialization."""
        mock_settings.DSPY_MODEL = "test-model"
        mock_settings.CEREBRAS_API_KEY = "test-key"
        mock_settings.CEREBRAS_BASE_URL = "test-url"
        
        optimizer = DSPyOptimizer()
        
        assert optimizer.config is not None
        assert optimizer.evaluator is not None
        
        # Verify DSPy was configured
        mock_dspy.configure.assert_called_once()
    
    @patch('dealgraph.reasoning.dspy.optimizer.dspy')
    @patch('dealgraph.reasoning.dspy.optimizer.settings')
    @patch('dealgraph.reasoning.dspy.optimizer.load_prompt')
    def test_optimize_prompt_mock(self, mock_load_prompt, mock_settings, mock_dspy):
        """Test prompt optimization with mocked DSPy."""
        mock_settings.DSPY_MODEL = "test-model"
        mock_settings.CEREBRAS_API_KEY = "test-key"
        mock_settings.CEREBRAS_BASE_URL = "test-url"
        
        # Mock prompt loading
        mock_load_prompt.return_value = {
            "content": "Test prompt template {query} {deals_block}",
            "metadata": {"description": "Test system prompt"}
        }
        
        # Mock optimization result
        mock_optimized_program = Mock()
        mock_optimized_program.get.side_effect = [
            "Optimized system prompt",
            "Optimized user prompt"
        ]
        
        # Mock teleprompter
        mock_teleprompter = Mock()
        mock_teleprompter.compile.return_value = mock_optimized_program
        mock_dspy.teleprompter.MIPRO.return_value = mock_teleprompter
        
        optimizer = DSPyOptimizer()
        
        # Test optimization
        result = optimizer.optimize_deal_reasoner_prompt(
            baseline_prompt_version="v1",
            num_candidate_prompts=5,
            max_evaluations=50
        )
        
        assert "baseline_version" in result
        assert "optimized_version" in result
        assert "improvement_score" in result
        assert "meets_threshold" in result
        
        # Verify optimization was called
        mock_teleprompter.compile.assert_called_once()
    
    @patch('dealgraph.reasoning.dspy.optimizer.DSPyOptimizer')
    def test_get_set_optimizer(self, mock_optimizer_class):
        """Test global optimizer get/set functions."""
        # Set a mock optimizer
        mock_optimizer = Mock()
        set_dspy_optimizer(mock_optimizer)
        
        # Get should return the same instance
        retrieved_optimizer = get_dspy_optimizer()
        assert retrieved_optimizer == mock_optimizer
    
    @patch('dealgraph.reasoning.dspy.optimizer.DSPyOptimizer')
    @patch('dealgraph.reasoning.dspy.optimizer.load_prompt')
    def test_save_optimized_prompt(self, mock_load_prompt, mock_optimizer_class):
        """Test saving optimized prompt."""
        mock_optimizer_instance = Mock()
        mock_optimizer_class.return_value = mock_optimizer_instance
        
        optimization_result = {
            "optimized_version": "v2",
            "optimization_time": "2024-11-23T10:00:00",
            "improvement_score": 0.15,
            "baseline_version": "v1",
            "candidate_prompts_evaluated": 10,
            "total_evaluations": 100,
            "system_prompt": "Optimized system prompt",
            "user_prompt": "Optimized user prompt",
            "meets_threshold": True
        }
        
        # Mock path creation
        with patch('dealgraph.reasoning.dspy.optimizer.Path') as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = mock_path.return_value
            
            # Mock file writing
            with patch('builtins.open', mock_open()) as mock_file:
                saved_path = mock_optimizer_instance.save_optimized_prompt(
                    optimization_result, "test/prompt.txt"
                )
                
                # Verify file was written
                mock_file.assert_called_once()
                assert "test/prompt.txt" in str(mock_file.return_value)


class TestDSPyIntegration:
    """Test DSPy integration with reasoner."""
    
    @patch('dealgraph.reasoning.reasoner.get_dspy_optimizer')
    def test_dspy_optimize_prompt(self, mock_get_optimizer):
        """Test dspy_optimize_prompt function."""
        mock_optimizer = Mock()
        mock_optimization_result = {
            "baseline_version": "v1",
            "optimized_version": "v2",
            "improvement_score": 0.12,
            "meets_threshold": True
        }
        mock_optimizer.optimize_deal_reasoner_prompt.return_value = mock_optimization_result
        mock_optimizer.save_optimized_prompt.return_value = "prompts/deal_reasoner/v2_optimized.txt"
        mock_get_optimizer.return_value = mock_optimizer
        
        result = dspy_optimize_prompt(
            baseline_version="v1",
            num_candidate_prompts=8,
            max_evaluations=80
        )
        
        assert result == mock_optimization_result
        mock_optimizer.optimize_deal_reasoner_prompt.assert_called_once()
        mock_optimizer.save_optimized_prompt.assert_called_once()
    
    @patch('dealgraph.reasoning.reasoner.get_dspy_optimizer')
    def test_dspy_evaluate_performance(self, mock_get_optimizer):
        """Test dspy_evaluate_performance function."""
        mock_optimizer = Mock()
        mock_evaluator = Mock()
        mock_evaluator.evaluate_reasoning_output.return_value = {
            "precision_at_3": 0.8,
            "playbook_quality": 0.7,
            "narrative_coherence": 0.6,
            "composite_score": 0.7
        }
        mock_optimizer.evaluator = mock_evaluator
        mock_get_optimizer.return_value = mock_optimizer
        
        reasoning_output = {
            "precedents": [{"deal_id": "test", "name": "Test", "similarity_reason": "Test"}],
            "playbook_levers": ["lever"],
            "risk_themes": ["risk"],
            "narrative_summary": "Test"
        }
        
        metrics = dspy_evaluate_performance(reasoning_output)
        
        assert metrics["composite_score"] == 0.7
        mock_evaluator.evaluate_reasoning_output.assert_called_once()
    
    def test_compare_reasoning_prompts(self):
        """Test compare_reasoning_prompts function."""
        with patch('dealgraph.reasoning.reasoner.deal_reasoner') as mock_deal_reasoner, \
             patch('dealgraph.reasoning.reasoner.extract_reasoning_metrics') as mock_metrics:
            
            # Mock reasoning results
            mock_metrics.side_effect = [
                {"num_precedents": 1, "has_precedents": True},
                {"num_precedents": 2, "has_precedents": True}
            ]
            
            # Mock test data
            query = "test query"
            ranked_deals = []
            prompt_versions = ["v1", "v2"]
            
            results = compare_reasoning_prompts(query, ranked_deals, prompt_versions)
            
            assert "v1" in results
            assert "v2" in results
            assert "v2_improvements" in results
            
            # Verify both prompts were tested
            assert mock_deal_reasoner.call_count == 2
    
    @patch('dealgraph.reasoning.reasoner.get_dspy_optimizer')
    def test_dspy_get_optimization_history(self, mock_get_optimizer):
        """Test dspy_get_optimization_history function."""
        mock_optimizer = Mock()
        mock_history = [
            {"baseline_version": "v1", "improvement_score": 0.1},
            {"baseline_version": "v1", "improvement_score": 0.15}
        ]
        mock_optimizer.get_optimization_history.return_value = mock_history
        mock_get_optimizer.return_value = mock_optimizer
        
        history = dspy_get_optimization_history()
        
        assert len(history) == 2
        assert history == mock_history
        mock_optimizer.get_optimization_history.assert_called_once()
    
    @patch('dealgraph.reasoning.reasoner.get_dspy_optimizer')
    def test_dspy_rollback_to_baseline(self, mock_get_optimizer):
        """Test dspy_rollback_to_baseline function."""
        mock_optimizer = Mock()
        mock_optimizer.rollback_to_baseline.return_value = True
        mock_get_optimizer.return_value = mock_optimizer
        
        result = dspy_rollback_to_baseline("v1")
        
        assert result is True
        mock_optimizer.rollback_to_baseline.assert_called_once_with("v1")
    
    def test_dspy_optimize_prompt_error(self):
        """Test error handling in dspy_optimize_prompt."""
        with patch('dealgraph.reasoning.reasoner.get_dspy_optimizer') as mock_get_optimizer:
            mock_optimizer = Mock()
            mock_optimizer.optimize_deal_reasoner_prompt.side_effect = Exception("Test error")
            mock_get_optimizer.return_value = mock_optimizer
            
            with pytest.raises(ReasoningError) as exc_info:
                dspy_optimize_prompt()
            
            assert "DSPy optimization failed: Test error" in str(exc_info.value)


class TestDSPyWorkflow:
    """Test complete DSPy optimization workflow."""
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow."""
        # This is a demonstration test that would need real DSPy setup
        # For now, just test that the functions are callable
        config = DSPyConfig(
            num_candidate_prompts=5,
            max_evaluations=50,
            min_improvement_threshold=0.1
        )
        
        evaluator = PerformanceEvaluator(config=config)
        optimizer = DSPyOptimizer(config=config, evaluator=evaluator)
        
        # Test that configuration is properly set
        assert optimizer.config.num_candidate_prompts == 5
        assert optimizer.config.max_evaluations == 50
        assert optimizer.config.min_improvement_threshold == 0.1
        
        # Test evaluation works
        reasoning_output = {
            "precedents": [{"deal_id": "test", "name": "Test", "similarity_reason": "Test"}],
            "playbook_levers": ["lever"],
            "risk_themes": ["risk"],
            "narrative_summary": "Test"
        }
        
        metrics = evaluator.evaluate_reasoning_output(reasoning_output)
        assert isinstance(metrics, dict)
        assert len(metrics) == 4  # 4 metrics calculated
