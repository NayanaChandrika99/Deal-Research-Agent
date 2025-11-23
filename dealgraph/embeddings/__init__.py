"""Embedding generation and vector indexing."""

import logging
import os

from .encoder import EmbeddingEncoder
from .index import DealEmbeddingIndex, EmbeddingIndexFactory
from .local_encoder import HashingEmbeddingEncoder

logger = logging.getLogger(__name__)
_LOCAL_MODE = os.getenv("USE_LOCAL_EMBEDDINGS", "auto").lower()


def get_embedding_encoder(force_local: bool | None = None):
    """
    Create the default embedding encoder.

    Args:
        force_local: When True, always use the hashing encoder. When False, always
            use the remote encoder. When None (default), attempt the remote encoder
            and fall back to the hashing encoder on failure.
    """
    if force_local is None:
        if _LOCAL_MODE == "true":
            force_local = True
        elif _LOCAL_MODE == "false":
            force_local = False

    if force_local is True:
        return HashingEmbeddingEncoder()

    if force_local is False:
        return EmbeddingEncoder()

    try:
        return EmbeddingEncoder()
    except Exception as exc:
        logger.warning("Falling back to hashing encoder: %s", exc)
        return HashingEmbeddingEncoder()


__all__ = [
    "EmbeddingEncoder",
    "HashingEmbeddingEncoder",
    "DealEmbeddingIndex",
    "EmbeddingIndexFactory",
    "get_embedding_encoder",
]
