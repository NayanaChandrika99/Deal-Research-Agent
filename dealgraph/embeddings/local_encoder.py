# ABOUTME: Deterministic hashing-based embedding encoder for offline scenarios.
# ABOUTME: Generates stable pseudo-embeddings so semantic search works without external APIs.

from __future__ import annotations

import hashlib
import math
import re
from typing import List

import numpy as np


class HashingEmbeddingEncoder:
    """
    Lightweight embedding encoder that deterministically hashes input text into vectors.

    This encoder is used as a fallback when remote embedding providers (e.g., OpenAI)
    are unavailable. It approximates semantic behavior by hashing tokens into a fixed
    number of buckets and normalizing the resulting vector.
    """

    _token_pattern = re.compile(r"[A-Za-z0-9]+")

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple text strings deterministically."""
        vectors: List[np.ndarray] = []
        for text in texts:
            tokens = self._tokenize(text)
            vector = np.zeros(self.dimension, dtype=np.float32)

            if not tokens:
                # Ensure zero-vector inputs still return usable embeddings.
                vector.fill(1.0 / math.sqrt(self.dimension))
            else:
                for token in tokens:
                    digest = hashlib.sha256(token.encode("utf-8")).digest()
                    # Use digest chunks to increment multiple buckets for each token.
                    for offset in range(0, len(digest), 4):
                        bucket = int.from_bytes(digest[offset:offset + 4], "big") % self.dimension
                        vector[bucket] += 1.0

                norm = np.linalg.norm(vector)
                if norm:
                    vector /= norm

            vectors.append(vector)

        return vectors

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into alphanumeric tokens."""
        tokens = self._token_pattern.findall(text.lower())
        if not tokens and text:
            tokens = [text.lower()]
        return tokens

    def get_dimension(self) -> int:
        """Return the embedding dimensionality."""
        return self.dimension

    def __repr__(self) -> str:
        return f"HashingEmbeddingEncoder(dim={self.dimension})"
