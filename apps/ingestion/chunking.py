"""
Semantic chunking with LangChain's RecursiveCharacterTextSplitter.
Supports configurable chunk size and overlap; adds positional metadata.
"""
from __future__ import annotations
from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


def chunk_text(
    text: str,
    metadata: dict | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping chunks suitable for embedding.

    Args:
        text:          Full document text.
        metadata:      Base metadata dict attached to every chunk.
        chunk_size:    Target token/char size per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of dicts: {chunk_index, content, metadata}.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    docs = splitter.create_documents([text], metadatas=[metadata or {}])

    return [
        {
            "chunk_index": idx,
            "content": doc.page_content,
            "metadata": {**doc.metadata, "chunk_index": idx},
        }
        for idx, doc in enumerate(docs)
    ]
