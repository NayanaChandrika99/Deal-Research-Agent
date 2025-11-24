"""Unit tests for reasoning module functionality."""

import pytest
import json
from unittest.mock import Mock, patch

from dealgraph.reasoning.prompts import PromptRegistry, load_prompt, DEAL_REASONER_NAIVE_PROMPT
from dealgraph.reasoning.llm_client import LLMClient, LLMClientError, get_llm_client
from dealgraph.reasoning.reasoner import (
    deal_reasoner,
    analyze_deals_with_naive_prompt,
    _format_deals_for_prompt,
    _parse_reasoning_output,
    validate_reasoning_output,
    extract_reasoning_metrics,
    ReasoningError
)
from dealgraph.data.schemas import Deal, Snippet, CandidateDeal, RankedDeal, Precedent, DealReasoningOutput


class TestPromptRegistry:
    """Test PromptRegistry functionality."""
    
    def test_load_prompt_with_cache(self, tmp_path):
        """Test loading prompts with caching."""
        # Create mock prompt structure
        prompt_dir = tmp_path / "deal_reasoner"
        prompt_dir.mkdir()
        
        prompt_file = prompt_dir / "v1_naive.txt"
        with open(prompt_file, 'w') as f:
            f.write("Test prompt content")
        
        metadata_file = prompt_dir / "v1_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "version": "v1",
                "name": "test_prompt",
                "description": "Test prompt description",
                "model": "test-model"
            }, f)
        
        registry = PromptRegistry(tmp_path)
        
        # First load should read from file
        result1 = registry.load_prompt("deal_reasoner", "v1")
        assert result1["content"] == "Test prompt content"
        assert result1["version"] == "v1"
        
        # Second load should use cache
        result2 = registry.load_prompt("deal_reasoner", "v1")
        assert result2["content"] == "Test prompt content"
        assert result2["version"] == "v1"
    
    def test_list_versions(self, tmp_path):
        """Test listing available versions."""
        prompt_dir = tmp_path / "deal_reasoner"
        prompt_dir.mkdir()
        
        # Create multiple versions
        versions = ["v1", "v2", "v1.0"]
        for version in versions:
            (prompt_dir / f"{version}_naive.txt").touch()
        
        registry = PromptRegistry(tmp_path)
        available_versions = registry.list_versions("deal_reasoner")
        
        # Should return sorted versions
        assert len(available_versions) == 3
        assert "v1" in available_versions
        assert "v2" in available_versions
        assert "v1.0" in available_versions
    
    def test_get_latest_version(self, tmp_path):
        """Test getting latest version."""
        prompt_dir = tmp_path / "deal_reasoner"
        prompt_dir.mkdir()
        
        # Create versions with semantic ordering
        (prompt_dir / "v1_naive.txt").touch()
        (prompt_dir / "v2_naive.txt").touch()
        (prompt_dir / "v1.2_naive.txt").touch()
        
        registry = PromptRegistry(tmp_path)
        latest = registry._get_latest_version("deal_reasoner")
        
        # Should return latest version
        assert latest == "v2"
    
    def test_load_nonexistent_prompt(self, tmp_path):
        """Test error when prompt doesn't exist."""
        registry = PromptRegistry(tmp_path)
        
        with pytest.raises(FileNotFoundError):
            registry.load_prompt("nonexistent")


class TestLLMClient:
    """Test LLMClient functionality."""
    
    def test_initialization(self):
        """Test client initialization."""
        with patch('dealgraph.config.settings.settings') as mock_settings:
            mock_settings.DSPY_MODEL = "test-model"
            mock_settings.CEREBRAS_API_KEY = "test-key"
            mock_settings.CEREBRAS_BASE_URL = "test-url"
            
            client = LLMClient()
            
            assert client.model == "test-model"
            assert client.api_key == "test-key"
            assert client.base_url == "test-url"
    
    def test_complete_json_success(self):
        """Test successful JSON completion."""
        with patch('openai.OpenAI') as MockOpenAI:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"test": "response"}'
            mock_client.chat.completions.create.return_value = mock_response
            
            MockOpenAI.return_value = mock_client
            
            client = LLMClient(api_key="test-key", base_url="test-url")
            result = client.complete_json("system", "user")
            
            assert result == {"test": "response"}
            mock_client.chat.completions.create.assert_called_once()
    
    def test_complete_json_invalid_json(self):
        """Test handling of invalid JSON response."""
        with patch('openai.OpenAI') as MockOpenAI:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "invalid json"
            mock_client.chat.completions.create.return_value = mock_response
            
            MockOpenAI.return_value = mock_client
            
            client = LLMClient(api_key="test-key", base_url="test-url")
            
            with pytest.raises(LLMClientError):
                client.complete_json("system", "user")
    
    def test_validate_json_response(self):
        """Test JSON response validation."""
        client = LLMClient(api_key="test-key", base_url="test-url")
        
        # Valid response
        valid_response = {
            "precedents": [{"deal_id": "test", "name": "Test"}],
            "playbook_levers": ["lever1"],
            "risk_themes": ["risk1"],
            "narrative_summary": "summary"
        }
        
        assert client.validate_json_response(valid_response, [
            "precedents", "playbook_levers", "risk_themes", "narrative_summary"
        ]) is True
        
        # Invalid response (missing field)
        invalid_response = {
            "precedents": [{"deal_id": "test"}],
            "playbook_levers": ["lever1"]
            # Missing risk_themes and narrative_summary
        }
        
        assert client.validate_json_response(invalid_response, [
            "precedents", "risk_themes"
        ]) is False


class TestReasoner:
    """Test deal reasoning functionality."""
    
    def test_reasoner_error_alias(self):
        """ReasoningError should be an alias for DealReasonerError."""
        from dealgraph.reasoning.reasoner import DealReasonerError, ReasoningError

        assert ReasoningError is DealReasonerError

    def test_format_deals_for_prompt(self):
        """Test formatting deals for prompt."""
        deals = [
            RankedDeal(
                candidate=CandidateDeal(
                    deal=Deal(
                        id="test1",
                        name="Test Deal 1",
                        sector_id="tech",
                        region_id="us",
                        is_platform=True,
                        status="current",
                        description="Test company 1"
                    ),
                    snippets=[
                        Snippet(id="snip1", deal_id="test1", source="news", text="Test snippet")
                    ],
                    text_similarity=0.9,
                    graph_features={"sector_match": 1}
                ),
                score=0.9,
                rank=1
            )
        ]
        
        formatted = _format_deals_for_prompt(deals)
        
        assert "Test Deal 1" in formatted
        assert "tech" in formatted
        assert "Test snippet" in formatted
        assert "0.900" in formatted  # similarity score
    
    def test_parse_reasoning_output_valid(self):
        """Test parsing valid reasoning output."""
        response = {
            "precedents": [
                {
                    "deal_id": "test1",
                    "name": "Test Deal",
                    "similarity_reason": "Strong sector match"
                }
            ],
            "playbook_levers": ["Operational improvements"],
            "risk_themes": ["Market competition"],
            "narrative_summary": "Test analysis summary"
        }
        
        deals = [
            RankedDeal(
                candidate=CandidateDeal(
                    deal=Deal(
                        id="test1",
                        name="Test Deal",
                        sector_id="tech",
                        region_id="us",
                        is_platform=True,
                        status="current",
                        description="Test"
                    ),
                    snippets=[],
                    text_similarity=0.9,
                    graph_features={}
                ),
                score=0.9,
                rank=1
            )
        ]
        
        result = _parse_reasoning_output(response, deals)
        
        assert len(result.precedents) == 1
        assert result.precedents[0].deal_id == "test1"
        assert result.precedents[0].similarity_reason == "Strong sector match"
        assert result.playbook_levers == ["Operational improvements"]
        assert result.risk_themes == ["Market competition"]
        assert result.narrative_summary == "Test analysis summary"
    
    def test_validate_reasoning_output(self):
        """Test reasoning output validation."""
        valid_output = DealReasoningOutput(
            precedents=[Precedent(deal_id="test", name="Test", similarity_reason="Test")],
            playbook_levers=["lever1"],
            risk_themes=["risk1"],
            narrative_summary="summary"
        )
        
        assert validate_reasoning_output(valid_output) is True
        
        # Invalid output (no content)
        empty_output = DealReasoningOutput(
            precedents=[],
            playbook_levers=[],
            risk_themes=[],
            narrative_summary=""
        )
        
        assert validate_reasoning_output(empty_output) is False
    
    def test_extract_reasoning_metrics(self):
        """Test extracting metrics from reasoning output."""
        output = DealReasoningOutput(
            precedents=[Precedent(deal_id="test", name="Test", similarity_reason="Test")],
            playbook_levers=["lever1", "lever2"],
            risk_themes=["risk1"],
            narrative_summary="This is a comprehensive analysis of the deal landscape with detailed insights."
        )
        
        metrics = extract_reasoning_metrics(output)
        
        assert metrics["num_precedents"] == 1
        assert metrics["num_playbook_levers"] == 2
        assert metrics["num_risk_themes"] == 1
        assert metrics["narrative_length"] > 0
        assert metrics["has_precedents"] is True
        assert metrics["has_playbook"] is True
        assert metrics["has_risks"] is True
        assert metrics["has_narrative"] is True


class TestIntegration:
    """Integration tests for reasoning functionality."""
    
    def test_reasoning_with_mock_llm(self):
        """Test complete reasoning workflow with mocked LLM."""
        with patch('dealgraph.reasoning.reasoner.get_llm_client') as mock_get_client:
            mock_client = Mock()
            mock_client.complete_json.return_value = {
                "precedents": [
                    {
                        "deal_id": "test1",
                        "name": "Test Deal",
                        "similarity_reason": "Strong match"
                    }
                ],
                "playbook_levers": ["Strategic acquisitions"],
                "risk_themes": ["Integration complexity"],
                "narrative_summary": "Strong precedent with strategic value"
            }
            mock_get_client.return_value = mock_client
            
            # Create test data
            deals = [
                RankedDeal(
                    candidate=CandidateDeal(
                        deal=Deal(
                            id="test1",
                            name="Test Deal",
                            sector_id="tech",
                            region_id="us",
                            is_platform=True,
                            status="current",
                            description="Test company"
                        ),
                        snippets=[],
                        text_similarity=0.9,
                        graph_features={}
                    ),
                    score=0.9,
                    rank=1
                )
            ]
            
            # Run reasoning
            result = deal_reasoner("test query", deals)
            
            # Verify result structure
            assert isinstance(result, DealReasoningOutput)
            assert len(result.precedents) == 1
            assert result.precedents[0].deal_id == "test1"
            assert len(result.playbook_levers) == 1
            assert len(result.risk_themes) == 1
            assert len(result.narrative_summary) > 0
            
            # Verify LLM was called
            mock_client.complete_json.assert_called_once()
    
    def test_naive_prompt_analysis(self):
        """Test analysis with naive prompt specifically."""
        with patch('dealgraph.reasoning.reasoner.get_llm_client') as mock_get_client:
            mock_client = Mock()
            mock_client.complete_json.return_value = {
                "precedents": [],
                "playbook_levers": [],
                "risk_themes": [],
                "narrative_summary": "No relevant precedents found"
            }
            mock_get_client.return_value = mock_client
            
            deals = []
            
            result = analyze_deals_with_naive_prompt("query", deals)
            
            assert isinstance(result, DealReasoningOutput)
            assert result.narrative_summary == "No relevant precedents found"
    
    def test_error_handling(self):
        """Test error handling in reasoning."""
        with patch('dealgraph.reasoning.reasoner.get_llm_client') as mock_get_client:
            mock_client = Mock()
            mock_client.complete_json.side_effect = LLMClientError("API error")
            mock_get_client.return_value = mock_client
            
            deals = []
            
            with pytest.raises(ReasoningError):
                deal_reasoner("query", deals)
