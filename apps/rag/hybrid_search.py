"""
Hybrid BM25 + dense vector search with score fusion (RRF).
BM25 runs in-memory over the retrieved candidate set; can be swapped
for Pinecone's native sparse-dense hybrid when using the serverless tier.
"""
from __future__ import annotations
import math
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from apps.api.core.logging import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def reciprocal_rank_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    Args:
        dense_docs:  Documents ranked by vector similarity.
        sparse_docs: Documents ranked by BM25 score.
        k:           RRF constant (higher = less emphasis on top ranks).

    Returns:
        Re-ranked and deduplicated document list.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:128]
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content[:128]
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def hybrid_retrieve(
    query: str,
    dense_docs: list[Document],
    top_k: int = 6,
) -> list[Document]:
    """
    Re-rank dense retrieval results using BM25 on the candidate set.

    Args:
        query:      User query.
        dense_docs: Candidate documents from vector search.
        top_k:      Final number of documents to return.

    Returns:
        Fused and re-ranked document list (length <= top_k).
    """
    if not dense_docs:
        return []

    corpus = [_tokenize(d.page_content) for d in dense_docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))

    sparse_ranked = [
        dense_docs[i]
        for i in sorted(range(len(dense_docs)), key=lambda x: scores[x], reverse=True)
    ]

    fused = reciprocal_rank_fusion(dense_docs, sparse_ranked)
    return fused[:top_k]
