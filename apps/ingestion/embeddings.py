"""
Embedding generation using sentence-transformers (BGE-large-en).
Singleton model loaded once; supports batch processing.
"""
from __future__ import annotations
from functools import lru_cache

from sentence_transformers import SentenceTransformer
import numpy as np

from apps.api.core.config import get_settings
from apps.api.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("loading_embedding_model", model=settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a list of text strings.

    Args:
        texts:      List of strings to embed.
        batch_size: Batch size for the encoder.

    Returns:
        List of embedding vectors (list of floats).
    """
    model = _get_model()
    # BGE models benefit from a query prefix during retrieval; for indexing we use as-is
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string with the BGE instruction prefix.

    Returns:
        Single embedding vector.
    """
    model = _get_model()
    # BGE models recommend this prefix for retrieval queries
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vec: np.ndarray = model.encode(
        [prefixed],
        normalize_embeddings=True,
    )
    return vec[0].tolist()
