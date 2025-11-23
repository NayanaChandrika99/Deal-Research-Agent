"""Unit tests for embeddings index functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from dealgraph.embeddings.index import DealEmbeddingIndex, EmbeddingIndexFactory
from dealgraph.embeddings import EmbeddingEncoder


class TestDealEmbeddingIndex:
    """Test DealEmbeddingIndex class functionality."""
    
    def test_initialization(self):
        """Test DealEmbeddingIndex initializes correctly."""
        index = DealEmbeddingIndex(embedding_dim=1536)
        
        assert index.embedding_dim == 1536
        assert index.encoder is None
        assert index.size() == 0
        assert index.is_empty() is True
        assert len(index._deal_ids) == 0
        assert index._embeddings.shape == (0, 1536)
        assert index._capacity is None
    
    def test_initialization_with_capacity(self):
        """Test DealEmbeddingIndex with capacity limit."""
        index = DealEmbeddingIndex(embedding_dim=768, capacity=100)
        
        assert index.embedding_dim == 768
        assert index._capacity == 100
    
    def test_add_single_embedding(self):
        """Test adding a single embedding."""
        index = DealEmbeddingIndex(embedding_dim=512)
        
        vector = np.random.rand(512).astype(np.float32)
        deal_id = "deal_001"
        metadata = {"name": "Test Deal", "sector": "tech"}
        
        index.add(deal_id, vector, metadata)
        
        assert index.size() == 1
        assert index.contains(deal_id) is True
        assert not index.is_empty()
        
        retrieved_vector = index.get_vector(deal_id)
        np.testing.assert_array_equal(retrieved_vector, vector)
        
        retrieved_metadata = index.get_metadata(deal_id)
        assert retrieved_metadata == metadata
    
    def test_add_multiple_embeddings(self):
        """Test adding multiple embeddings."""
        index = DealEmbeddingIndex(embedding_dim=256)
        
        vectors = []
        for i in range(5):
            vector = np.random.rand(256).astype(np.float32)
            deal_id = f"deal_{i:03d}"
            metadata = {"index": i}
            
            index.add(deal_id, vector, metadata)
            vectors.append(vector)
        
        assert index.size() == 5
        
        # Verify all embeddings are stored correctly
        for i in range(5):
            deal_id = f"deal_{i:03d}"
            retrieved_vector = index.get_vector(deal_id)
            np.testing.assert_array_equal(retrieved_vector, vectors[i])
            
            metadata = index.get_metadata(deal_id)
            assert metadata["index"] == i
    
    def test_add_invalid_vector_dimension(self):
        """Test error when vector dimension doesn't match."""
        index = DealEmbeddingIndex(embedding_dim=512)
        
        vector = np.random.rand(256)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Vector dimension .* doesn't match expected 512"):
            index.add("deal_001", vector)
    
    def test_add_duplicate_deal_id(self):
        """Test error when adding duplicate deal ID."""
        index = DealEmbeddingIndex(embedding_dim=512)
        
        vector = np.random.rand(512).astype(np.float32)
        index.add("deal_001", vector)
        
        with pytest.raises(ValueError, match="Deal ID 'deal_001' already exists"):
            index.add("deal_001", vector)
    
    def test_add_exceeds_capacity(self):
        """Test error when exceeding capacity."""
        index = DealEmbeddingIndex(embedding_dim=512, capacity=2)
        
        vector = np.random.rand(512).astype(np.float32)
        index.add("deal_001", vector)
        index.add("deal_002", vector)
        
        with pytest.raises(ValueError, match="Index capacity .* exceeded"):
            index.add("deal_003", vector)
    
    def test_search_empty_index(self):
        """Test searching empty index returns empty results."""
        index = DealEmbeddingIndex(embedding_dim=512)
        
        query_vector = np.random.rand(512).astype(np.float32)
        results = index.search(query_vector)
        
        assert results == []
    
    def test_search_single_item(self):
        """Test search with single item in index."""
        index = DealEmbeddingIndex(embedding_dim=3)
        
        # Add a vector at [1, 0, 0]
        stored_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        index.add("deal_001", stored_vector)
        
        # Search with same vector
        query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query_vector, top_k=1)
        
        assert len(results) == 1
        assert results[0][0] == "deal_001"
        assert abs(results[0][1] - 1.0) < 1e-6  # Perfect similarity
    
    def test_search_multiple_items(self):
        """Test search with multiple items."""
        index = DealEmbeddingIndex(embedding_dim=3)
        
        # Add three vectors
        vectors = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # deal_001
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # deal_002  
            np.array([0.0, 0.0, 1.0], dtype=np.float32),  # deal_003
        ]
        
        for i, vector in enumerate(vectors):
            index.add(f"deal_{i+1:03d}", vector)
        
        # Search with vector most similar to [1, 0, 0]
        query_vector = np.array([1.0, 0.1, 0.1], dtype=np.float32)
        results = index.search(query_vector, top_k=3)
        
        assert len(results) == 3
        assert results[0][0] == "deal_001"  # Most similar
        assert results[1][0] in ["deal_002", "deal_003"]  # Less similar
        
        # Verify similarity scores are descending
        similarities = [result[1] for result in results]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_search_with_filter(self):
        """Test search with filtered deal IDs."""
        index = DealEmbeddingIndex(embedding_dim=3)
        
        # Add vectors
        vectors = [np.array([1.0, 0.0, 0.0], dtype=np.float32) for _ in range(5)]
        for i in range(5):
            index.add(f"deal_{i+1:03d}", vectors[i])
        
        # Search with filter
        filter_ids = ["deal_001", "deal_003", "deal_005"]
        query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query_vector, filter_ids=filter_ids)
        
        # Should only return results from filtered IDs
        result_ids = [result[0] for result in results]
        assert set(result_ids).issubset(set(filter_ids))
        assert len(results) == 3
    
    def test_search_without_distances(self):
        """Test search without returning distance scores."""
        index = DealEmbeddingIndex(embedding_dim=3)
        
        vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        index.add("deal_001", vector)
        
        query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query_vector, include_distances=False)
        
        assert len(results) == 1
        assert results[0][0] == "deal_001"
        assert results[0][1] == 1.0  # Default score
    
    def test_search_invalid_query_dimension(self):
        """Test error with invalid query vector dimension."""
        index = DealEmbeddingIndex(embedding_dim=512)
        
        vector = np.random.rand(256).astype(np.float32)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Query vector dimension .* doesn't match expected 512"):
            index.search(vector)
    
    def test_get_vector_nonexistent(self):
        """Test retrieving vector for nonexistent deal."""
        index = DealEmbeddingIndex(embedding_dim=512)
        
        vector = index.get_vector("nonexistent")
        assert vector is None
    
    def test_remove_existing(self):
        """Test removing existing deal."""
        index = DealEmbeddingIndex(embedding_dim=3)
        
        vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        index.add("deal_001", vector)
        
        result = index.remove("deal_001")
        assert result is True
        assert index.size() == 0
        assert index.contains("deal_001") is False
        assert index.is_empty()
    
    def test_remove_nonexistent(self):
        """Test removing nonexistent deal."""
        index = DealEmbeddingIndex(embedding_dim=512)
        
        result = index.remove("nonexistent")
        assert result is False
    
    def test_batch_add(self):
        """Test batch adding multiple embeddings."""
        index = DealEmbeddingIndex(embedding_dim=256)
        
        deal_ids = ["deal_001", "deal_002", "deal_003"]
        vectors = np.random.rand(3, 256).astype(np.float32)
        metadata_list = [{"name": f"Deal {i}"} for i in range(3)]
        
        index.batch_add(deal_ids, vectors, metadata_list)
        
        assert index.size() == 3
        
        # Verify all embeddings
        for i, deal_id in enumerate(deal_ids):
            retrieved_vector = index.get_vector(deal_id)
            np.testing.assert_array_equal(retrieved_vector, vectors[i])
            
            metadata = index.get_metadata(deal_id)
            assert metadata["name"] == f"Deal {i}"
    
    def test_batch_add_mismatched_dimensions(self):
        """Test error with mismatched batch dimensions."""
        index = DealEmbeddingIndex(embedding_dim=256)
        
        deal_ids = ["deal_001", "deal_002"]
        vectors = np.random.rand(1, 256).astype(np.float32)  # Wrong number of vectors
        
        with pytest.raises(ValueError, match="Mismatch: .* IDs but .* vectors"):
            index.batch_add(deal_ids, vectors)
    
    def test_batch_add_duplicate_ids(self):
        """Test error with duplicate IDs in batch."""
        index = DealEmbeddingIndex(embedding_dim=256)
        
        # Add one deal first
        vector = np.random.rand(256).astype(np.float32)
        index.add("deal_001", vector)
        
        # Try to add duplicate in batch
        deal_ids = ["deal_001", "deal_002"]  # Duplicate
        vectors = np.random.rand(2, 256).astype(np.float32)
        
        with pytest.raises(ValueError, match="Deal ID 'deal_001' already exists"):
            index.batch_add(deal_ids, vectors)
    
    def test_get_stats_empty(self):
        """Test getting stats for empty index."""
        index = DealEmbeddingIndex(embedding_dim=512)
        
        stats = index.get_stats()
        
        assert stats["size"] == 0
        assert stats["embedding_dim"] == 512
        assert stats["capacity"] is None
        assert stats["is_empty"] is True
        assert stats["memory_usage_mb"] == 0.0
        assert "avg_similarity" not in stats  # No similarities for empty index
    
    def test_get_stats_with_data(self):
        """Test getting stats with embeddings."""
        index = DealEmbeddingIndex(embedding_dim=3, capacity=10)
        
        # Add embeddings
        vectors = [np.array([i, 0, 0], dtype=np.float32) for i in range(5)]
        for i, vector in enumerate(vectors):
            index.add(f"deal_{i+1:03d}", vector)
        
        stats = index.get_stats()
        
        assert stats["size"] == 5
        assert stats["embedding_dim"] == 3
        assert stats["capacity"] == 10
        assert stats["is_empty"] is False
        assert stats["memory_usage_mb"] > 0
        
        # Similarity statistics should be present
        assert "avg_similarity" in stats
        assert "max_similarity" in stats
        assert "min_similarity" in stats
        assert "std_similarity" in stats
    
    def test_clear(self):
        """Test clearing the index."""
        index = DealEmbeddingIndex(embedding_dim=256)
        
        # Add some data
        vector = np.random.rand(256).astype(np.float32)
        index.add("deal_001", vector)
        index.add("deal_002", vector)
        
        assert index.size() == 2
        
        # Clear the index
        index.clear()
        
        assert index.size() == 0
        assert index.is_empty()
        assert index._embeddings.shape == (0, 256)
        assert len(index._deal_ids) == 0
        assert len(index._id_to_index) == 0
    
    def test_contains(self):
        """Test the 'in' operator."""
        index = DealEmbeddingIndex(embedding_dim=256)
        
        vector = np.random.rand(256).astype(np.float32)
        index.add("deal_001", vector)
        
        assert "deal_001" in index
        assert "deal_002" not in index
    
    def test_len(self):
        """Test the len() function."""
        index = DealEmbeddingIndex(embedding_dim=256)
        
        assert len(index) == 0
        
        vector = np.random.rand(256).astype(np.float32)
        index.add("deal_001", vector)
        
        assert len(index) == 1
        
        index.add("deal_002", vector)
        assert len(index) == 2
    
    def test_repr(self):
        """Test string representation."""
        index = DealEmbeddingIndex(embedding_dim=512, capacity=100)
        
        expected = "DealEmbeddingIndex(size=0, dim=512, capacity=100)"
        assert repr(index) == expected
        
        # Add some data
        vector = np.random.rand(512).astype(np.float32)
        index.add("deal_001", vector)
        
        expected = "DealEmbeddingIndex(size=1, dim=512, capacity=100)"
        assert repr(index) == expected


class TestEmbeddingIndexFactory:
    """Test EmbeddingIndexFactory class."""
    
    @patch('dealgraph.embeddings.encoder.EmbeddingEncoder')
    def test_create_with_encoder(self, mock_encoder_class):
        """Test creating index with encoder."""
        mock_encoder = Mock()
        mock_encoder.get_dimension.return_value = 1536
        
        index = EmbeddingIndexFactory.create_with_encoder(mock_encoder, capacity=100)
        
        assert isinstance(index, DealEmbeddingIndex)
        assert index.encoder == mock_encoder
        assert index.embedding_dim == 1536
        assert index._capacity == 100
    
    @patch('dealgraph.embeddings.index.EmbeddingEncoder')
    def test_create_with_settings(self, mock_encoder_class):
        """Test creating index with settings."""
        # Mock the encoder instance
        mock_encoder_instance = Mock()
        mock_encoder_instance.get_dimension.return_value = 1536
        mock_encoder_class.return_value = mock_encoder_instance
        
        # Mock settings
        with patch('dealgraph.config.settings.settings') as mock_settings:
            mock_settings.EMBEDDING_DIMENSION = 1536
            
            index = EmbeddingIndexFactory.create_with_settings(capacity=50)
            
            assert isinstance(index, DealEmbeddingIndex)
            assert index.embedding_dim == 1536
            assert index._capacity == 50
            mock_encoder_class.assert_called_once()


class TestIntegration:
    """Integration tests with real sample data."""
    
    def test_full_workflow(self):
        """Test complete workflow: encode → store → search."""
        # This test requires valid API access, so we'll mock it
        from unittest.mock import patch
        
        with patch.object(EmbeddingEncoder, 'embed_text') as mock_embed:
            # Mock embedding response
            mock_vector = np.random.rand(1536).astype(np.float32)
            mock_embed.return_value = mock_vector
            
            # Create encoder and index
            encoder = EmbeddingEncoder()
            index = DealEmbeddingIndex(1536)
            
            # Encode and store some deals
            deals = [
                ("tech_001", "CloudTech Solutions - Enterprise cloud platform"),
                ("health_001", "MedCare Plus - Healthcare management services"),
                ("ind_001", "Industrial Group - Manufacturing services platform"),
            ]
            
            for deal_id, text in deals:
                vector = encoder.embed_text(text)
                index.add(deal_id, vector, {"description": text})
            
            # Verify storage
            assert index.size() == 3
            
            # Search for similar deals
            query = "Healthcare technology services"
            query_vector = encoder.embed_text(query)
            results = index.search(query_vector, top_k=3)
            
            # Should find healthcare-related deals
            result_ids = [result[0] for result in results]
            assert "health_001" in result_ids
            
            # Verify we can retrieve vectors and metadata
            for deal_id in result_ids:
                vector = index.get_vector(deal_id)
                metadata = index.get_metadata(deal_id)
                
                assert vector is not None
                assert vector.shape == (1536,)
                assert metadata is not None
                assert "description" in metadata
    
    def test_performance_with_larger_dataset(self):
        """Test performance with larger dataset."""
        index = DealEmbeddingIndex(embedding_dim=768)
        
        # Add 100 random embeddings
        n_deals = 100
        vectors = np.random.rand(n_deals, 768).astype(np.float32)
        
        # Use batch add for efficiency
        deal_ids = [f"deal_{i:03d}" for i in range(n_deals)]
        index.batch_add(deal_ids, vectors)
        
        assert index.size() == n_deals
        
        # Test search performance
        query_vector = np.random.rand(768).astype(np.float32)
        results = index.search(query_vector, top_k=10)
        
        assert len(results) == 10
        
        # Verify search results are sorted by similarity
        similarities = [result[1] for result in results]
        assert similarities == sorted(similarities, reverse=True)
        
        # Test getting stats
        stats = index.get_stats()
        assert stats["size"] == n_deals
        assert stats["embedding_dim"] == 768
        assert stats["memory_usage_mb"] > 0
