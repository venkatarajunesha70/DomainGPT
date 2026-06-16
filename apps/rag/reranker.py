"""
Cross-encoder reranker using FlashRank (lightweight, no GPU required).
Runs after hybrid retrieval to select the most relevant chunks.
"""
from __future__ import annotations
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

from apps.api.core.logging import get_logger

logger = get_logger(__name__)

# FlashRank ms-marco-MiniLM model – fast CPU reranker
_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
    return _ranker


def rerank(query: str, docs: list[Document], top_n: int = 4) -> list[Document]:
    """
    Rerank documents with a cross-encoder and return the top_n.

    Args:
        query:  User query string.
        docs:   Candidate documents from hybrid retrieval.
        top_n:  How many documents to keep after reranking.

    Returns:
        Reranked documents (most relevant first).
    """
    if not docs:
        return []

    ranker = _get_ranker()
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    reranked = []
    for res in results[:top_n]:
        doc = docs[res["id"]]
        doc.metadata["rerank_score"] = res["score"]
        reranked.append(doc)

    logger.debug("reranked", kept=len(reranked), total=len(docs))
    return reranked
