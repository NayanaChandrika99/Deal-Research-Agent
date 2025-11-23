"""Unit tests for graph search functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from dealgraph.retrieval.graph_search import (
    graph_semantic_search,
    rank_candidates,
    batch_search,
    search_with_explanations,
    get_search_explanation,
    build_snippets_index,
    validate_search_parameters
)
from dealgraph.data.schemas import Deal, Snippet, CandidateDeal
from dealgraph.embeddings import EmbeddingEncoder, DealEmbeddingIndex
from dealgraph.data.graph_builder import DealGraph


class TestGraphSemanticSearch:
    """Test graph_semantic_search function."""
    
    def test_basic_search_functionality(self):
        """Test basic search without filters."""
        # Mock encoder
        encoder = Mock(spec=EmbeddingEncoder)
        encoder.embed_text.return_value = np.random.rand(1536).astype(np.float32)
        
        # Mock embedding index
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_index.search.return_value = [
            ('deal_001', 0.9),
            ('deal_002', 0.8),
            ('deal_003', 0.7)
        ]
        
        # Mock deal graph
        deal_graph = Mock(spec=DealGraph)
        deal_graph.get_deal_neighbors.return_value = {
            'sectors': [{'id': 'tech'}],
            'regions': [{'id': 'us'}],
            'events': [],
            'snippets': [],
            'related_deals': [],
            'addons': [],
            'platforms': []
        }
        deal_graph.graph.has_node.return_value = True
        deal_graph.graph.out_degree.return_value = 2
        deal_graph.graph.in_degree.return_value = 1
        
        # Test data
        deals = [
            Deal(id="deal_001", name="Tech Corp", sector_id="tech", region_id="us",
                 is_platform=True, status="current", description="Technology company"),
            Deal(id="deal_002", name="Health Corp", sector_id="healthcare", region_id="us",
                 is_platform=False, status="current", description="Healthcare company"),
            Deal(id="deal_003", name="Industrial Co", sector_id="industrial", region_id="us",
                 is_platform=True, status="current", description="Industrial company")
        ]
        
        snippets_by_deal = {
            'deal_001': [Snippet(id="snip_001", deal_id="deal_001", source="news", text="Tech news")],
            'deal_002': [Snippet(id="snip_002", deal_id="deal_002", source="news", text="Health news")],
            'deal_003': [Snippet(id="snip_003", deal_id="deal_003", source="news", text="Industrial news")]
        }
        
        # Perform search
        results = graph_semantic_search(
            query="technology company",
            encoder=encoder,
            deal_index=deal_index,
            deal_graph=deal_graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal,
            top_k=3
        )
        
        # Verify results
        assert len(results) == 3
        
        # Check first result has highest similarity
        assert results[0].deal.id == "deal_001"
        assert results[0].text_similarity == 0.9
        
        # Verify all results have required fields
        for candidate in results:
            assert isinstance(candidate, CandidateDeal)
            assert candidate.deal.id in ['deal_001', 'deal_002', 'deal_003']
            assert candidate.text_similarity > 0
            assert 'sector_match' in candidate.graph_features
            assert 'region_match' in candidate.graph_features
    
    def test_search_with_sector_filter(self):
        """Test search with sector filtering."""
        encoder = Mock(spec=EmbeddingEncoder)
        encoder.embed_text.return_value = np.random.rand(1536).astype(np.float32)
        
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_index.search.return_value = [
            ('deal_001', 0.9),  # tech
            ('deal_002', 0.8),  # healthcare
            ('deal_003', 0.7)   # industrial
        ]
        
        deal_graph = Mock(spec=DealGraph)
        deal_graph.get_deal_neighbors.return_value = {
            'sectors': [{'id': 'tech'}],
            'regions': [{'id': 'us'}],
            'events': [],
            'snippets': [],
            'related_deals': [],
            'addons': [],
            'platforms': []
        }
        deal_graph.graph.has_node.return_value = True
        deal_graph.graph.out_degree.return_value = 2
        deal_graph.graph.in_degree.return_value = 1
        
        deals = [
            Deal(id="deal_001", name="Tech Corp", sector_id="tech", region_id="us",
                 is_platform=True, status="current", description="Technology"),
            Deal(id="deal_002", name="Health Corp", sector_id="healthcare", region_id="us",
                 is_platform=False, status="current", description="Healthcare"),
            Deal(id="deal_003", name="Industrial Co", sector_id="industrial", region_id="us",
                 is_platform=True, status="current", description="Industrial")
        ]
        
        snippets_by_deal = {}
        
        # Search with tech sector filter
        results = graph_semantic_search(
            query="company",
            encoder=encoder,
            deal_index=deal_index,
            deal_graph=deal_graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal,
            filter_sectors=['tech', 'healthcare']
        )
        
        # Should only return deals in tech or healthcare
        result_ids = [c.deal.id for c in results]
        assert 'deal_001' in result_ids  # tech
        assert 'deal_002' in result_ids  # healthcare
        assert 'deal_003' not in result_ids  # industrial filtered out
    
    def test_search_with_platform_filter(self):
        """Test search with platform/add-on filtering."""
        encoder = Mock(spec=EmbeddingEncoder)
        encoder.embed_text.return_value = np.random.rand(1536).astype(np.float32)
        
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_index.search.return_value = [
            ('deal_001', 0.9),  # platform
            ('deal_002', 0.8),  # addon
            ('deal_003', 0.7)   # platform
        ]
        
        deal_graph = Mock(spec=DealGraph)
        deal_graph.get_deal_neighbors.return_value = {
            'sectors': [{'id': 'tech'}],
            'regions': [{'id': 'us'}],
            'events': [],
            'snippets': [],
            'related_deals': [],
            'addons': [],
            'platforms': []
        }
        deal_graph.graph.has_node.return_value = True
        deal_graph.graph.out_degree.return_value = 2
        deal_graph.graph.in_degree.return_value = 1
        
        deals = [
            Deal(id="deal_001", name="Platform Corp", sector_id="tech", region_id="us",
                 is_platform=True, status="current", description="Platform"),
            Deal(id="deal_002", name="Addon Corp", sector_id="tech", region_id="us",
                 is_platform=False, status="current", description="Addon"),
            Deal(id="deal_003", name="Another Platform", sector_id="tech", region_id="us",
                 is_platform=True, status="current", description="Another Platform")
        ]
        
        snippets_by_deal = {}
        
        # Search for platforms only
        results = graph_semantic_search(
            query="platform",
            encoder=encoder,
            deal_index=deal_index,
            deal_graph=deal_graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal,
            filter_platforms=True
        )
        
        # Should only return platforms
        for candidate in results:
            assert candidate.deal.is_platform is True
    
    def test_search_with_min_similarity(self):
        """Test search with minimum similarity threshold."""
        encoder = Mock(spec=EmbeddingEncoder)
        encoder.embed_text.return_value = np.random.rand(1536).astype(np.float32)
        
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_index.search.return_value = [
            ('deal_001', 0.95),  # High similarity
            ('deal_002', 0.8),   # Above threshold
            ('deal_003', 0.6),   # Below threshold
            ('deal_004', 0.4)    # Below threshold
        ]
        
        deal_graph = Mock(spec=DealGraph)
        deal_graph.get_deal_neighbors.return_value = {
            'sectors': [{'id': 'tech'}],
            'regions': [{'id': 'us'}],
            'events': [],
            'snippets': [],
            'related_deals': [],
            'addons': [],
            'platforms': []
        }
        deal_graph.graph.has_node.return_value = True
        deal_graph.graph.out_degree.return_value = 2
        deal_graph.graph.in_degree.return_value = 1
        
        deals = [
            Deal(id="deal_001", name="High Similarity", sector_id="tech", region_id="us",
                 is_platform=True, status="current", description="High similarity"),
            Deal(id="deal_002", name="Medium Similarity", sector_id="tech", region_id="us",
                 is_platform=False, status="current", description="Medium similarity"),
            Deal(id="deal_003", name="Low Similarity", sector_id="tech", region_id="us",
                 is_platform=True, status="current", description="Low similarity"),
            Deal(id="deal_004", name="Very Low Similarity", sector_id="tech", region_id="us",
                 is_platform=False, status="current", description="Very low similarity")
        ]
        
        snippets_by_deal = {}
        
        # Search with minimum similarity of 0.7
        results = graph_semantic_search(
            query="test",
            encoder=encoder,
            deal_index=deal_index,
            deal_graph=deal_graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal,
            min_similarity=0.7
        )
        
        # Should only return deals with similarity >= 0.7
        result_ids = [c.deal.id for c in results]
        assert 'deal_001' in result_ids  # 0.95 >= 0.7
        assert 'deal_002' in result_ids  # 0.8 >= 0.7
        assert 'deal_003' not in result_ids  # 0.6 < 0.7
        assert 'deal_004' not in result_ids  # 0.4 < 0.7
    
    def test_empty_search_results(self):
        """Test behavior when search returns no results."""
        encoder = Mock(spec=EmbeddingEncoder)
        encoder.embed_text.return_value = np.random.rand(1536).astype(np.float32)
        
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_index.search.return_value = []  # No results
        
        deal_graph = Mock(spec=DealGraph)
        
        deals = []
        snippets_by_deal = {}
        
        results = graph_semantic_search(
            query="test",
            encoder=encoder,
            deal_index=deal_index,
            deal_graph=deal_graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal
        )
        
        assert results == []
    
    def test_search_with_no_matching_deals(self):
        """Test behavior when no deals match filtering criteria."""
        encoder = Mock(spec=EmbeddingEncoder)
        encoder.embed_text.return_value = np.random.rand(1536).astype(np.float32)
        
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_index.search.return_value = [
            ('deal_001', 0.9),
            ('deal_002', 0.8)
        ]
        
        deal_graph = Mock(spec=DealGraph)
        deal_graph.get_deal_neighbors.return_value = {
            'sectors': [{'id': 'tech'}],
            'regions': [{'id': 'us'}],
            'events': [],
            'snippets': [],
            'related_deals': [],
            'addons': [],
            'platforms': []
        }
        deal_graph.graph.has_node.return_value = True
        deal_graph.graph.out_degree.return_value = 2
        deal_graph.graph.in_degree.return_value = 1
        
        # Deals don't match the search results
        deals = [
            Deal(id="deal_003", name="Different Corp", sector_id="healthcare", region_id="europe",
                 is_platform=True, status="current", description="Different")
        ]
        
        snippets_by_deal = {}
        
        # Search with very restrictive filters
        results = graph_semantic_search(
            query="test",
            encoder=encoder,
            deal_index=deal_index,
            deal_graph=deal_graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal,
            filter_sectors=['nonexistent'],
            filter_regions=['nonexistent']
        )
        
        assert results == []
    
    def test_search_encoding_error(self):
        """Test handling of encoding errors."""
        encoder = Mock(spec=EmbeddingEncoder)
        encoder.embed_text.side_effect = Exception("API error")
        
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_graph = Mock(spec=DealGraph)
        deals = []
        snippets_by_deal = {}
        
        with pytest.raises(ValueError, match="Failed to encode query"):
            graph_semantic_search(
                query="test",
                encoder=encoder,
                deal_index=deal_index,
                deal_graph=deal_graph,
                deals=deals,
                snippets_by_deal=snippets_by_deal
            )


class TestRankCandidates:
    """Test rank_candidates function."""
    
    def test_ranking_by_relevance(self):
        """Test that candidates are ranked by relevance score."""
        # Create mock candidates with different features
        deal1 = Deal(id="deal_001", name="High Match", sector_id="tech", region_id="us",
                    is_platform=True, status="current", description="High match")
        deal2 = Deal(id="deal_002", name="Medium Match", sector_id="tech", region_id="us",
                    is_platform=False, status="current", description="Medium match")
        deal3 = Deal(id="deal_003", name="Low Match", sector_id="healthcare", region_id="europe",
                    is_platform=True, status="current", description="Low match")
        
        candidates = [
            CandidateDeal(
                deal=deal1,
                snippets=[],
                text_similarity=0.9,
                graph_features={
                    'sector_match': 1,
                    'region_match': 1,
                    'is_platform': 1,
                    'num_addons': 2,
                    'has_exit': 0,
                    'text_graph_alignment': 0.85
                }
            ),
            CandidateDeal(
                deal=deal2,
                snippets=[],
                text_similarity=0.8,
                graph_features={
                    'sector_match': 1,
                    'region_match': 0,
                    'is_platform': 0,
                    'num_addons': 0,
                    'has_exit': 0,
                    'text_graph_alignment': 0.6
                }
            ),
            CandidateDeal(
                deal=deal3,
                snippets=[],
                text_similarity=0.7,
                graph_features={
                    'sector_match': 0,
                    'region_match': 0,
                    'is_platform': 1,
                    'num_addons': 1,
                    'has_exit': 1,
                    'text_graph_alignment': 0.4
                }
            )
        ]
        
        ranked = rank_candidates(candidates)
        
        # Should be ordered by relevance (first should have highest score)
        assert ranked[0].deal.id == "deal_001"  # High text similarity + sector match + platform
        assert ranked[1].deal.id == "deal_002"  # Medium text similarity + sector match
        assert ranked[2].deal.id == "deal_003"  # Lower text similarity, no matches
    
    def test_single_candidate(self):
        """Test ranking with single candidate."""
        deal = Deal(id="deal_001", name="Only Deal", sector_id="tech", region_id="us",
                   is_platform=True, status="current", description="Only deal")
        
        candidate = CandidateDeal(
            deal=deal,
            snippets=[],
            text_similarity=0.5,
            graph_features={'sector_match': 1}
        )
        
        ranked = rank_candidates([candidate])
        
        assert len(ranked) == 1
        assert ranked[0].deal.id == "deal_001"
    
    def test_empty_candidates(self):
        """Test ranking with empty candidate list."""
        ranked = rank_candidates([])
        
        assert ranked == []


class TestBatchSearch:
    """Test batch_search function."""
    
    def test_batch_search_multiple_queries(self):
        """Test batch search for multiple queries."""
        encoder = Mock(spec=EmbeddingEncoder)
        encoder.embed_text.return_value = np.random.rand(1536).astype(np.float32)
        
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_index.search.side_effect = [
            [('deal_001', 0.9)],  # Query 1 results
            [('deal_002', 0.8)],  # Query 2 results
            []  # Query 3 results (no matches)
        ]
        
        deal_graph = Mock(spec=DealGraph)
        deal_graph.get_deal_neighbors.return_value = {
            'sectors': [{'id': 'tech'}],
            'regions': [{'id': 'us'}],
            'events': [],
            'snippets': [],
            'related_deals': [],
            'addons': [],
            'platforms': []
        }
        deal_graph.graph.has_node.return_value = True
        deal_graph.graph.out_degree.return_value = 2
        deal_graph.graph.in_degree.return_value = 1
        
        deals = [
            Deal(id="deal_001", name="Tech Corp", sector_id="tech", region_id="us",
                 is_platform=True, status="current", description="Tech"),
            Deal(id="deal_002", name="Health Corp", sector_id="healthcare", region_id="us",
                 is_platform=False, status="current", description="Health")
        ]
        
        snippets_by_deal = {}
        
        queries = ["technology", "healthcare", "manufacturing"]
        results = batch_search(
            queries=queries,
            encoder=encoder,
            deal_index=deal_index,
            deal_graph=deal_graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal,
            top_k=5
        )
        
        # Should return results for each query
        assert len(results) == 3
        
        # First query should have results
        assert len(results[0]) == 1
        assert results[0][0].deal.id == "deal_001"
        
        # Second query should have results
        assert len(results[1]) == 1
        assert results[1][0].deal.id == "deal_002"
        
        # Third query should have no results
        assert len(results[2]) == 0
    
    def test_batch_search_with_error(self):
        """Test batch search handles errors gracefully."""
        encoder = Mock(spec=EmbeddingEncoder)
        
        # First call succeeds, second call fails
        encoder.embed_text.side_effect = [
            np.random.rand(1536).astype(np.float32),  # Success
            Exception("API error")  # Error
        ]
        
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_index.search.return_value = [('deal_001', 0.9)]
        
        deal_graph = Mock(spec=DealGraph)
        deal_graph.get_deal_neighbors.return_value = {
            'sectors': [{'id': 'tech'}],
            'regions': [{'id': 'us'}],
            'events': [],
            'snippets': [],
            'related_deals': [],
            'addons': [],
            'platforms': []
        }
        deal_graph.graph.has_node.return_value = True
        deal_graph.graph.out_degree.return_value = 2
        deal_graph.graph.in_degree.return_value = 1
        
        deals = [
            Deal(id="deal_001", name="Tech Corp", sector_id="tech", region_id="us",
                 is_platform=True, status="current", description="Tech")
        ]
        
        snippets_by_deal = {}
        
        queries = ["success", "error"]
        results = batch_search(
            queries=queries,
            encoder=encoder,
            deal_index=deal_index,
            deal_graph=deal_graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal
        )
        
        # First query should succeed
        assert len(results[0]) == 1
        
        # Second query should return empty (error handled)
        assert len(results[1]) == 0


class TestSearchExplanations:
    """Test search explanation functionality."""
    
    def test_get_search_explanation(self):
        """Test generation of search explanation."""
        deal = Deal(id="deal_001", name="Tech Platform", sector_id="tech", region_id="us",
                   is_platform=True, status="current", description="Technology platform")
        
        candidate = CandidateDeal(
            deal=deal,
            snippets=[],
            text_similarity=0.85,
            graph_features={
                'sector_match': 1,
                'region_match': 1,
                'is_platform': 1,
                'num_addons': 3,
                'has_exit': 0,
                'text_graph_alignment': 0.8
            }
        )
        
        query = "technology platform in united states"
        
        explanation = get_search_explanation(candidate, query)
        
        assert explanation['deal_id'] == "deal_001"
        assert explanation['deal_name'] == "Tech Platform"
        assert explanation['query'] == query
        assert explanation['text_similarity'] == 0.85
        assert 'relevance_factors' in explanation
        
        # Check that relevant factors are mentioned
        factors = explanation['relevance_factors']
        assert any("High text similarity" in factor for factor in factors)
        assert any("Sector match" in factor for factor in factors)
        assert any("Region match" in factor for factor in factors)
        assert any("Platform deal" in factor for factor in factors)
        assert any("add-on acquisitions" in factor for factor in factors)
    
    def test_search_with_explanations(self):
        """Test search with explanations returns both results and explanations."""
        encoder = Mock(spec=EmbeddingEncoder)
        encoder.embed_text.return_value = np.random.rand(1536).astype(np.float32)
        
        deal_index = Mock(spec=DealEmbeddingIndex)
        deal_index.search.return_value = [('deal_001', 0.9)]
        
        deal_graph = Mock(spec=DealGraph)
        deal_graph.get_deal_neighbors.return_value = {
            'sectors': [{'id': 'tech'}],
            'regions': [{'id': 'us'}],
            'events': [],
            'snippets': [],
            'related_deals': [],
            'addons': [],
            'platforms': []
        }
        deal_graph.graph.has_node.return_value = True
        deal_graph.graph.out_degree.return_value = 2
        deal_graph.graph.in_degree.return_value = 1
        
        deals = [
            Deal(id="deal_001", name="Tech Corp", sector_id="tech", region_id="us",
                 is_platform=True, status="current", description="Technology company")
        ]
        
        snippets_by_deal = {}
        
        results, explanations = search_with_explanations(
            query="technology company",
            encoder=encoder,
            deal_index=deal_index,
            deal_graph=deal_graph,
            deals=deals,
            snippets_by_deal=snippets_by_deal
        )
        
        assert len(results) == 1
        assert len(explanations) == 1
        assert explanations[0]['deal_id'] == "deal_001"


class TestSnippetsIndex:
    """Test snippets indexing functionality."""
    
    def test_build_snippets_index(self):
        """Test building index from snippets list."""
        snippets = [
            Snippet(id="snip_001", deal_id="deal_001", source="news", text="Deal 1 news"),
            Snippet(id="snip_002", deal_id="deal_001", source="case_study", text="Deal 1 case study"),
            Snippet(id="snip_003", deal_id="deal_002", source="news", text="Deal 2 news"),
            Snippet(id="snip_004", deal_id="deal_003", source="news", text="Deal 3 news")
        ]
        
        index = build_snippets_index(snippets)
        
        assert len(index) == 3
        assert "deal_001" in index
        assert "deal_002" in index
        assert "deal_003" in index
        
        # Check snippets for deal_001
        deal_001_snippets = index["deal_001"]
        assert len(deal_001_snippets) == 2
        assert deal_001_snippets[0].id == "snip_001"
        assert deal_001_snippets[1].id == "snip_002"
        
        # Check single snippets
        assert len(index["deal_002"]) == 1
        assert index["deal_002"][0].id == "snip_003"
        assert len(index["deal_003"]) == 1
        assert index["deal_003"][0].id == "snip_004"
    
    def test_empty_snippets(self):
        """Test behavior with empty snippets list."""
        index = build_snippets_index([])
        
        assert index == {}
    
    def test_single_snippet(self):
        """Test with single snippet."""
        snippets = [
            Snippet(id="snip_001", deal_id="deal_001", source="news", text="Single snippet")
        ]
        
        index = build_snippets_index(snippets)
        
        assert len(index) == 1
        assert "deal_001" in index
        assert len(index["deal_001"]) == 1


class TestParameterValidation:
    """Test parameter validation functionality."""
    
    def test_valid_parameters(self):
        """Test validation of valid parameters."""
        valid_params = {
            'top_k': 10,
            'min_similarity': 0.5,
            'filter_sectors': ['tech', 'healthcare'],
            'filter_regions': ['us', 'europe'],
            'filter_platforms': True
        }
        
        # Should not raise any exception
        validate_search_parameters(**valid_params)
    
    def test_invalid_top_k(self):
        """Test validation failure for invalid top_k."""
        invalid_params = {'top_k': 0}  # Too small
        
        with pytest.raises(ValueError, match="top_k must be an integer between 1 and 1000"):
            validate_search_parameters(**invalid_params)
        
        invalid_params = {'top_k': 1500}  # Too large
        
        with pytest.raises(ValueError, match="top_k must be an integer between 1 and 1000"):
            validate_search_parameters(**invalid_params)
        
        invalid_params = {'top_k': "10"}  # Wrong type
        
        with pytest.raises(ValueError, match="top_k must be an integer between 1 and 1000"):
            validate_search_parameters(**invalid_params)
    
    def test_invalid_min_similarity(self):
        """Test validation failure for invalid min_similarity."""
        invalid_params = {'min_similarity': -0.1}  # Below range
        
        with pytest.raises(ValueError, match="min_similarity must be a float between 0.0 and 1.0"):
            validate_search_parameters(**invalid_params)
        
        invalid_params = {'min_similarity': 1.5}  # Above range
        
        with pytest.raises(ValueError, match="min_similarity must be a float between 0.0 and 1.0"):
            validate_search_parameters(**invalid_params)
        
        invalid_params = {'min_similarity': "0.5"}  # Wrong type
        
        with pytest.raises(ValueError, match="min_similarity must be a float between 0.0 and 1.0"):
            validate_search_parameters(**invalid_params)
    
    def test_invalid_filter_lists(self):
        """Test validation failure for invalid filter lists."""
        # Non-list filter
        with pytest.raises(ValueError, match="filter_sectors must be a list"):
            validate_search_parameters(filter_sectors="tech")
        
        # List with non-string items
        with pytest.raises(ValueError, match="All items in filter_sectors must be strings"):
            validate_search_parameters(filter_sectors=[1, 2, 3])
        
        # Non-list filter regions
        with pytest.raises(ValueError, match="filter_regions must be a list"):
            validate_search_parameters(filter_regions="us")
        
        # List with non-string items
        with pytest.raises(ValueError, match="All items in filter_regions must be strings"):
            validate_search_parameters(filter_regions=["us", 123])
