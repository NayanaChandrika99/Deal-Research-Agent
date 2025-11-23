# ABOUTME: Wrapper around OpenAI embeddings API for text vectorization.
# ABOUTME: Converts text strings into embedding vectors for semantic search.

from typing import List, Union
import numpy as np
from openai import OpenAI
from ..config.settings import settings


class EmbeddingEncoder:
    """
    Wrapper for OpenAI embeddings API.
    
    Uses text-embedding-3-small by default for cost-effective, high-quality embeddings.
    """
    
    def __init__(
        self, 
        model: str = None,
        api_key: str = None,
        dimensions: int = None
    ):
        """
        Initialize the embedding encoder.
        
        Args:
            model: OpenAI embedding model name (defaults to settings.EMBEDDING_MODEL)
            api_key: OpenAI API key (defaults to settings.OPENAI_API_KEY)
            dimensions: Output embedding dimensions (defaults to settings.EMBEDDING_DIMENSION)
        """
        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.dimensions = dimensions or settings.EMBEDDING_DIMENSION
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple text strings in a single API call.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors as numpy arrays
            
        Note:
            - Maximum 2048 inputs per request
            - Maximum 8192 tokens per input
            - Maximum 300,000 tokens total across all inputs
        """
        if not texts:
            return []
        
        # Remove empty strings
        texts = [t.strip() for t in texts if t.strip()]
        
        if not texts:
            return []
        
        # Call OpenAI embeddings API
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        
        # Extract embeddings from response
        embeddings = [
            np.array(data.embedding, dtype=np.float32)
            for data in response.data
        ]
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self.dimensions
    
    def __repr__(self) -> str:
        return f"EmbeddingEncoder(model={self.model}, dimensions={self.dimensions})"

