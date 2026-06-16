"""
Pinecone-backed vector retriever.
Supports metadata filtering for multi-tenant isolation.
"""
from __future__ import annotations
from functools import lru_cache

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from apps.ingestion.embeddings import embed_query
from apps.api.core.config import get_settings
from apps.api.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class TenantEmbeddingWrapper:
    """LangChain-compatible embedding wrapper around our local model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from apps.ingestion.embeddings import embed_texts
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)


@lru_cache(maxsize=1)
def _get_pinecone_store() -> PineconeVectorStore:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    return PineconeVectorStore(
        index=index,
        embedding=TenantEmbeddingWrapper(),
        text_key="content",
    )


def retrieve(
    query: str,
    tenant_id: str,
    top_k: int = 6,
    filter_metadata: dict | None = None,
) -> list[Document]:
    """
    Retrieve top-k relevant document chunks for the query.

    Args:
        query:           User question / rewritten query.
        tenant_id:       Pinecone namespace / metadata filter for isolation.
        top_k:           Number of chunks to return.
        filter_metadata: Additional metadata filters (e.g. document_id).

    Returns:
        List of LangChain Document objects with page_content and metadata.
    """
    store = _get_pinecone_store()
    metadata_filter = {"tenant_id": {"$eq": tenant_id}}
    if filter_metadata:
        metadata_filter.update(filter_metadata)

    docs = store.similarity_search(
        query=query,
        k=top_k,
        filter=metadata_filter,
        namespace=tenant_id,
    )
    logger.debug("retrieved_chunks", count=len(docs), tenant=tenant_id)
    return docs


def upsert_chunks(
    chunks: list[dict],
    tenant_id: str,
) -> None:
    """
    Upsert pre-chunked documents into Pinecone.

    Args:
        chunks:    List of {chunk_id, content, metadata, embedding}.
        tenant_id: Pinecone namespace.
    """
    from pinecone import Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    vectors = [
        (
            chunk["chunk_id"],
            chunk["embedding"],
            {**chunk["metadata"], "content": chunk["content"]},
        )
        for chunk in chunks
    ]
    # Batch upsert in groups of 100
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i : i + 100], namespace=tenant_id)
    logger.info("upserted_vectors", count=len(vectors), tenant=tenant_id)
