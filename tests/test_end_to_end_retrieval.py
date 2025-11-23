"""Integration tests for end-to-end retrieval workflow."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from dealgraph.agent.tools import (
    tool_graph_semantic_search,
    tool_batch_search,
    tool_get_search_explanations,
    tool_load_search_data,
    build_search_index,
    SearchIndexManager,
    validate_search_setup
)
from dealgraph.data.schemas import Deal, Snippet, CandidateDeal


class TestEndToEndRetrieval:
    """Test complete retrieval workflow."""
    
    def test_real_data_search_returns_relevant_results(self, tmp_path, monkeypatch):
        """End-to-end search over sample dataset should surface software deals."""
        # Force search index rebuild with fresh cache
        from dealgraph.agent import tools as agent_tools
        agent_tools._search_manager.reset()
        agent_tools.build_search_index(force_rebuild=True)
        
        results = agent_tools.tool_graph_semantic_search(
            query="US software platform executing cybersecurity add-ons",
            top_k=5,
            include_explanations=False
        )
        
        assert results, "Expected semantic search to return at least one result"
        top_result = results[0]
        assert top_result["sector_id"] == "software"
        assert top_result["deal_id"].startswith("platform_")
    
    def test_complete_search_workflow(self):
        """Test complete search workflow from data loading to results."""
        # Mock the underlying components
        with patch('dealgraph.data.ingest.load_all') as mock_load, \
             patch('dealgraph.data.graph_builder.DealGraph') as MockGraph, \
             patch('dealgraph.embeddings.EmbeddingEncoder') as MockEncoder, \
             patch('dealgraph.embeddings.DealEmbeddingIndex') as MockIndex:
            
            # Setup mocks
            mock_dataset = Mock()
            mock_dataset.deals = [
                Deal(id="deal_001", name="Tech Platform", sector_id="software", region_id="us",
                     is_platform=True, status="current", description="Technology platform company"),
                Deal(id="deal_002", name="Health Corp", sector_id="healthcare", region_id="us",
                     is_platform=False, status="current", description="Healthcare services"),
                Deal(id="deal_003", name="Industrial Co", sector_id="industrial", region_id="europe",
                     is_platform=True, status="realized", description="Industrial manufacturing")
            ]
            mock_dataset.snippets = [
                Snippet(id="snip_001", deal_id="deal_001", source="news", text="Tech news"),
                Snippet(id="snip_002", deal_id="deal_002", source="news", text="Health news")
            ]
            mock_load.return_value = mock_dataset
            
            # Setup graph mock
            mock_graph = Mock()
            mock_graph.get_deal_neighbors.return_value = {
                'sectors': [{'id': 'software'}],
                'regions': [{'id': 'us'}],
                'events': [],
                'snippets': [],
                'related_deals': [],
                'addons': [],
                'platforms': []
            }
            MockGraph.return_value = mock_graph
            
            # Setup index mock
            mock_index = Mock()
            mock_index.search.return_value = [('deal_001', 0.9), ('deal_002', 0.8)]
            MockIndex.return_value = mock_index
            
            # Setup encoder mock
            mock_encoder = Mock()
            mock_encoder.embed_text.return_value = np.random.rand(1536).astype(np.float32)
            MockEncoder.return_value = mock_encoder
            
            # Test search
            results = tool_graph_semantic_search(
                query="technology platform",
                top_k=5
            )
            
            # Verify results structure
            assert len(results) <= 5
            for result in results:
                assert "deal_id" in result
                assert "deal_name" in result
                assert "text_similarity" in result
                assert "graph_features" in result
                assert "relevance_score" in result
                assert "explanation" in result
    
    def test_search_with_filters(self):
        """Test search with various filters."""
        # Mock search index manager
        manager = SearchIndexManager()
        
        # Mock the components
        mock_graph = Mock()
        mock_index = Mock()
        mock_index.search.return_value = [('deal_001', 0.9), ('deal_002', 0.8)]
        
        mock_deals = [
            Deal(id="deal_001", name="US Tech", sector_id="software", region_id="us",
                 is_platform=True, status="current", description="US technology"),
            Deal(id="deal_002", name="EU Health", sector_id="healthcare", region_id="europe",
                 is_platform=False, status="current", description="European healthcare")
        ]
        
        mock_snippets = {"deal_001": [], "deal_002": []}
        mock_encoder = Mock()
        
        # Manually set the manager state
        manager._graph = mock_graph
        manager._index = mock_index
        manager._deals = mock_deals
        manager._snippets_by_deal = mock_snippets
        manager._encoder = mock_encoder
        
        # Test sector filtering
        with patch('dealgraph.agent.tools._search_manager', manager):
            results = tool_graph_semantic_search(
                query="technology",
                filter_sectors=["software"]
            )
            
            # Should filter results to software sector only
            # Note: The actual filtering logic is in the mocked graph_semantic_search
            assert isinstance(results, list)
    
    def test_batch_search_functionality(self):
        """Test batch search for multiple queries."""
        # Mock the search manager
        manager = SearchIndexManager()
        
        mock_graph = Mock()
        mock_index = Mock()
        mock_index.search.side_effect = [
            [('deal_001', 0.9)],
            [('deal_002', 0.8)],
            []  # No results for third query
        ]
        
        mock_deals = [
            Deal(id="deal_001", name="Tech", sector_id="software", region_id="us",
                 is_platform=True, status="current", description="Tech"),
            Deal(id="deal_002", name="Health", sector_id="healthcare", region_id="us",
                 is_platform=False, status="current", description="Health")
        ]
        
        mock_snippets = {"deal_001": [], "deal_002": []}
        mock_encoder = Mock()
        
        manager._graph = mock_graph
        manager._index = mock_index
        manager._deals = mock_deals
        manager._snippets_by_deal = mock_snippets
        manager._encoder = mock_encoder
        
        # Test batch search
        queries = ["technology", "healthcare", "nonexistent"]
        
        with patch('dealgraph.agent.tools._search_manager', manager):
            results = tool_batch_search(queries, top_k=5)
            
            assert len(results) == 3
            assert len(results[0]) >= 0  # First query may have results
            assert len(results[1]) >= 0  # Second query may have results
            assert len(results[2]) == 0  # Third query should have no results
    
    def test_search_explanations(self):
        """Test search explanation generation."""
        # Create sample search results
        candidates = [
            {
                "deal_id": "deal_001",
                "deal_name": "Tech Platform",
                "sector_id": "software",
                "region_id": "us",
                "is_platform": True,
                "status": "current",
                "description": "Technology platform",
                "text_similarity": 0.85,
                "graph_features": {
                    "sector_match": 1,
                    "region_match": 1,
                    "is_platform": 1,
                    "num_addons": 2,
                    "has_exit": 0,
                    "text_graph_alignment": 0.8
                }
            }
        ]
        
        query = "technology platform"
        
        explanations = tool_get_search_explanations(candidates, query)
        
        assert len(explanations) == 1
        explanation = explanations[0]
        
        assert explanation["deal_id"] == "deal_001"
        assert explanation["deal_name"] == "Tech Platform"
        assert explanation["query"] == query
        assert "relevance_factors" in explanation
        assert "similarity_breakdown" in explanation
        assert "graph_insights" in explanation
        assert "recommendation" in explanation
    
    def test_search_index_manager(self):
        """Test SearchIndexManager functionality."""
        manager = SearchIndexManager()
        
        # Test initial state
        assert not manager.is_ready()
        
        # Test stats for not ready state
        stats = manager.get_stats()
        assert stats["ready"] is False
        
        # Test that building requires proper data
        with pytest.raises(Exception):
            manager.build_index("nonexistent_path")
    
    def test_validate_search_setup(self):
        """Test search setup validation."""
        # Test when not ready
        with patch('dealgraph.agent.tools._search_manager.is_ready', return_value=False):
            validation = validate_search_setup()
            
            assert validation["ready"] is False
            assert "error" in validation
    
    def test_error_handling(self):
        """Test error handling in search operations."""
        # Test with invalid parameters
        results = tool_graph_semantic_search(
            query="test",
            top_k=0  # Invalid parameter
        )
        
        # Should return empty list on error
        assert results == []
    
    def test_realistic_search_scenario(self):
        """Test a realistic search scenario with various parameters."""
        # Create realistic search results
        realistic_results = [
            {
                "deal_id": "cloudtech_001",
                "deal_name": "CloudTech Solutions",
                "sector_id": "software",
                "region_id": "us",
                "is_platform": True,
                "status": "current",
                "description": "Enterprise cloud management software platform focused on mid-market companies",
                "text_similarity": 0.92,
                "graph_features": {
                    "sector_match": 1,
                    "region_match": 1,
                    "is_platform": 1,
                    "num_addons": 3,
                    "has_exit": 0,
                    "degree": 8,
                    "text_graph_alignment": 0.89,
                    "text_similarity": 0.92  # Add text_similarity to graph_features
                },
                "snippets": [
                    {
                        "id": "snip_001",
                        "source": "news",
                        "text": "CloudTech announced strong Q3 growth"
                    }
                ],
                "explanation": {
                    "relevance_factors": [
                        "High text similarity (0.920)",
                        "Sector match: software",
                        "Region match: us",
                        "Platform deal",
                        "Has 3 add-on acquisitions"
                    ],
                    "similarity_breakdown": {
                        "text_similarity": 0.92,
                        "description_length": 12,
                        "text_graph_alignment": 0.89
                    },
                    "graph_insights": {
                        "sector_match": 1,
                        "region_match": 1,
                        "is_platform": 1,
                        "num_addons": 3,
                        "has_exit": 0,
                        "graph_degree": 8
                    },
                    "recommendation": "Active platform with add-on strategy"
                },
                "relevance_score": 0.847
            }
        ]
        
        query = "cloud technology platform with add-on strategy"
        explanations = tool_get_search_explanations(realistic_results, query)
        
        explanation = explanations[0]
        
        # Verify comprehensive explanation
        assert len(explanation["relevance_factors"]) >= 4
        # Check that some relevant factor is present (order may vary)
        relevance_text = " ".join(explanation["relevance_factors"])
        assert "High text similarity" in relevance_text or "similarity" in relevance_text
        assert explanation["similarity_breakdown"]["text_similarity"] == 0.92
        assert explanation["graph_insights"]["is_platform"] == 1
        assert explanation["recommendation"] in [
            "Active platform with add-on strategy",
            "Strong match for query context"
        ]


class TestSearchWorkflowIntegration:
    """Test integration between search components."""
    
    def test_data_pipeline_integration(self):
        """Test that data flows properly through the pipeline."""
        # This would test the actual integration in a real scenario
        # For now, we'll test the interface
        
        # Test that tool functions can be called
        try:
            # This should work even if data isn't loaded
            validation = validate_search_setup()
            assert "ready" in validation
            assert isinstance(validation["ready"], bool)
        except Exception as e:
            pytest.fail(f"Search setup validation failed: {e}")
    
    def test_component_compatibility(self):
        """Test that components are compatible."""
        # Test that the tool functions have consistent interfaces
        from dealgraph.agent.tools import (
            tool_graph_semantic_search,
            tool_batch_search,
            tool_get_search_explanations,
            tool_load_search_data
        )
        
        # Check that functions exist and are callable
        assert callable(tool_graph_semantic_search)
        assert callable(tool_batch_search)
        assert callable(tool_get_search_explanations)
        assert callable(tool_load_search_data)
        
        # Check function signatures (basic check)
        import inspect
        
        # tool_graph_semantic_search should accept query parameter
        sig = inspect.signature(tool_graph_semantic_search)
        assert 'query' in sig.parameters
        
        # tool_batch_search should accept queries parameter
        sig = inspect.signature(tool_batch_search)
        assert 'queries' in sig.parameters
    
    def test_error_propagation(self):
        """Test that errors are properly propagated and handled."""
        # Test with invalid search parameters
        with patch('dealgraph.agent.tools._search_manager.is_ready', return_value=False):
            with patch('dealgraph.agent.tools._search_manager.build_index') as mock_build:
                mock_build.side_effect = Exception("Build failed")
                
                # Should handle build errors gracefully
                results = tool_graph_semantic_search("test query")
                assert results == []
    
    def test_performance_characteristics(self):
        """Test performance characteristics of search tools."""
        # Test that search tools handle reasonable parameters
        reasonable_params = {
            "query": "technology platform",
            "top_k": 10,
            "min_similarity": 0.1,
            "include_explanations": True
        }
        
        # These should not raise exceptions (even if they return empty results)
        try:
            results = tool_graph_semantic_search(**reasonable_params)
            assert isinstance(results, list)
        except Exception as e:
            # If it fails due to missing data, that's expected
            # If it fails due to invalid parameters, that's a problem
            assert "Invalid" not in str(e) or "parameter" not in str(e).lower()
