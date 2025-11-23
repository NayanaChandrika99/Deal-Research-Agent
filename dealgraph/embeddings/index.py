# ABOUTME: Vector storage and similarity search index for deal embeddings.
# ABOUTME: Provides efficient in-memory vector storage with cosine similarity search.

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .encoder import EmbeddingEncoder


class DealEmbeddingIndex:
    """
    In-memory vector index for deal embeddings with similarity search.
    
    Provides efficient storage and retrieval of deal embeddings using cosine similarity.
    Designed to work with the EmbeddingEncoder for consistent vector generation.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        encoder: Optional[EmbeddingEncoder] = None,
        capacity: Optional[int] = None
    ):
        """
        Initialize the embedding index.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            encoder: EmbeddingEncoder instance (for convenience methods)
            capacity: Maximum number of embeddings to store (None for unlimited)
        """
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        
        # Storage for embeddings and metadata
        self._deal_ids: List[str] = []
        self._embeddings: np.ndarray = np.zeros((0, embedding_dim), dtype=np.float32)
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._capacity = capacity
        
        # Index tracking
        self._id_to_index: Dict[str, int] = {}
    
    def add(self, deal_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an embedding to the index.
        
        Args:
            deal_id: Unique identifier for the deal
            vector: Embedding vector (must match embedding_dim)
            metadata: Optional metadata to store with the embedding
            
        Raises:
            ValueError: If vector dimension doesn't match or deal_id already exists
        """
        # Validate vector dimensions
        if vector.shape != (self.embedding_dim,):
            raise ValueError(f"Vector dimension {vector.shape} doesn't match expected {self.embedding_dim}")
        
        # Check for duplicate deal_id
        if deal_id in self._id_to_index:
            raise ValueError(f"Deal ID '{deal_id}' already exists in index")
        
        # Handle capacity
        if self._capacity and len(self._deal_ids) >= self._capacity:
            raise ValueError(f"Index capacity ({self._capacity}) exceeded")
        
        # Convert vector to float32 for memory efficiency
        vector = vector.astype(np.float32)
        
        # Add to storage
        if len(self._deal_ids) == 0:
            # First embedding
            self._embeddings = vector.reshape(1, -1)
        else:
            # Append to existing embeddings
            self._embeddings = np.vstack([self._embeddings, vector])
        
        self._deal_ids.append(deal_id)
        self._id_to_index[deal_id] = len(self._deal_ids) - 1
        
        # Store metadata
        self._metadata[deal_id] = metadata or {}
    
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 50,
        include_distances: bool = True,
        filter_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar embeddings using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            include_distances: Whether to include similarity scores in results
            filter_ids: Optional list of deal IDs to limit search to
            
        Returns:
            List of (deal_id, similarity_score) tuples, sorted by similarity (descending)
            
        Raises:
            ValueError: If query_vector dimension doesn't match
        """
        # Validate query vector
        if query_vector.shape != (self.embedding_dim,):
            raise ValueError(f"Query vector dimension {query_vector.shape} doesn't match expected {self.embedding_dim}")
        
        if len(self._deal_ids) == 0:
            return []
        
        # Convert query vector to float32
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # Filter embeddings if needed
        if filter_ids:
            filter_indices = [self._id_to_index[deal_id] for deal_id in filter_ids if deal_id in self._id_to_index]
            search_embeddings = self._embeddings[filter_indices]
            search_deal_ids = [self._deal_ids[i] for i in filter_indices]
        else:
            search_embeddings = self._embeddings
            search_deal_ids = self._deal_ids
        
        if len(search_deal_ids) == 0:
            return []
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, search_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
        
        # Build results
        results = []
        for idx in top_indices:
            deal_id = search_deal_ids[idx]
            similarity_score = float(similarities[idx])
            
            if include_distances:
                results.append((deal_id, similarity_score))
            else:
                results.append((deal_id, 1.0))  # Dummy score
        
        return results
    
    def get_vector(self, deal_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a stored embedding vector.
        
        Args:
            deal_id: Deal identifier to retrieve
            
        Returns:
            Embedding vector or None if not found
        """
        if deal_id not in self._id_to_index:
            return None
        
        index = self._id_to_index[deal_id]
        return self._embeddings[index].copy()
    
    def get_metadata(self, deal_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a deal.
        
        Args:
            deal_id: Deal identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self._metadata.get(deal_id)
    
    def remove(self, deal_id: str) -> bool:
        """
        Remove an embedding from the index.
        
        Args:
            deal_id: Deal identifier to remove
            
        Returns:
            True if deal was removed, False if not found
        """
        if deal_id not in self._id_to_index:
            return False
        
        index = self._id_to_index[deal_id]
        
        # Remove from embeddings array
        self._embeddings = np.delete(self._embeddings, index, axis=0)
        
        # Update tracking structures
        self._deal_ids.pop(index)
        self._id_to_index.pop(deal_id)
        self._metadata.pop(deal_id, None)
        
        # Update indices for remaining items
        for i in range(index, len(self._deal_ids)):
            deal_id_to_update = self._deal_ids[i]
            self._id_to_index[deal_id_to_update] = i
        
        return True
    
    def contains(self, deal_id: str) -> bool:
        """
        Check if a deal ID exists in the index.
        
        Args:
            deal_id: Deal identifier to check
            
        Returns:
            True if deal ID exists, False otherwise
        """
        return deal_id in self._id_to_index
    
    def size(self) -> int:
        """
        Get the number of embeddings in the index.
        
        Returns:
            Number of stored embeddings
        """
        return len(self._deal_ids)
    
    def is_empty(self) -> bool:
        """
        Check if the index is empty.
        
        Returns:
            True if index is empty, False otherwise
        """
        return len(self._deal_ids) == 0
    
    def get_all_ids(self) -> List[str]:
        """
        Get all stored deal IDs.
        
        Returns:
            List of all deal IDs in the index
        """
        return self._deal_ids.copy()
    
    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            Dimension of embedding vectors
        """
        return self.embedding_dim
    
    def clear(self) -> None:
        """
        Clear all embeddings from the index.
        """
        self._deal_ids.clear()
        self._embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
        self._metadata.clear()
        self._id_to_index.clear()
    
    def batch_add(
        self, 
        deal_ids: List[str], 
        vectors: np.ndarray, 
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add multiple embeddings in batch for efficiency.
        
        Args:
            deal_ids: List of deal identifiers
            vectors: 2D array of embedding vectors (n_items x embedding_dim)
            metadata_list: Optional list of metadata dictionaries
            
        Raises:
            ValueError: If dimensions don't match or deal IDs already exist
        """
        if len(deal_ids) != vectors.shape[0]:
            raise ValueError(f"Mismatch: {len(deal_ids)} IDs but {vectors.shape[0]} vectors")
        
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match expected {self.embedding_dim}")
        
        # Check for duplicate IDs
        for deal_id in deal_ids:
            if deal_id in self._id_to_index:
                raise ValueError(f"Deal ID '{deal_id}' already exists in index")
        
        # Check capacity
        if self._capacity and (len(self._deal_ids) + len(deal_ids)) > self._capacity:
            raise ValueError(f"Batch addition would exceed capacity ({self._capacity})")
        
        # Convert to float32
        vectors = vectors.astype(np.float32)
        
        # Store metadata
        if metadata_list is None:
            metadata_list = [None] * len(deal_ids)
        
        # Append to storage
        if len(self._deal_ids) == 0:
            self._embeddings = vectors
        else:
            self._embeddings = np.vstack([self._embeddings, vectors])
        
        # Update tracking
        start_index = len(self._deal_ids)
        for i, deal_id in enumerate(deal_ids):
            self._deal_ids.append(deal_id)
            self._id_to_index[deal_id] = start_index + i
            self._metadata[deal_id] = metadata_list[i] or {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        stats = {
            "size": self.size(),
            "embedding_dim": self.embedding_dim,
            "capacity": self._capacity,
            "is_empty": self.is_empty(),
            "memory_usage_mb": self._embeddings.nbytes / (1024 * 1024),
        }
        
        if not self.is_empty():
            # Calculate some similarity statistics
            sample_size = min(100, self.size())
            sample_indices = np.random.choice(self.size(), sample_size, replace=False)
            sample_embeddings = self._embeddings[sample_indices]
            
            # Compute pairwise similarities (sample only)
            similarities = cosine_similarity(sample_embeddings)
            
            # Remove diagonal (self-similarities)
            similarities = similarities[np.triu_indices_from(similarities, k=1)]
            
            stats.update({
                "avg_similarity": float(np.mean(similarities)),
                "max_similarity": float(np.max(similarities)),
                "min_similarity": float(np.min(similarities)),
                "std_similarity": float(np.std(similarities)),
            })
        
        return stats
    
    def __len__(self) -> int:
        """Return the number of embeddings in the index."""
        return self.size()
    
    def __contains__(self, deal_id: str) -> bool:
        """Check if deal_id is in the index."""
        return self.contains(deal_id)
    
    def __repr__(self) -> str:
        """String representation of the index."""
        return f"DealEmbeddingIndex(size={self.size()}, dim={self.embedding_dim}, capacity={self._capacity})"


class EmbeddingIndexFactory:
    """
    Factory class for creating and managing embedding indices.
    """
    
    @staticmethod
    def create_with_encoder(
        encoder: EmbeddingEncoder,
        capacity: Optional[int] = None
    ) -> DealEmbeddingIndex:
        """
        Create an embedding index using an existing encoder.
        
        Args:
            encoder: EmbeddingEncoder instance
            capacity: Optional capacity limit
            
        Returns:
            DealEmbeddingIndex instance
        """
        return DealEmbeddingIndex(
            embedding_dim=encoder.get_dimension(),
            encoder=encoder,
            capacity=capacity
        )
    
    @staticmethod
    def create_with_settings(capacity: Optional[int] = None) -> DealEmbeddingIndex:
        """
        Create an embedding index using application settings.
        
        Args:
            capacity: Optional capacity limit
            
        Returns:
            DealEmbeddingIndex instance
        """
        from ..config.settings import settings
        
        encoder = EmbeddingEncoder()
        return EmbeddingIndexFactory.create_with_encoder(encoder, capacity)
