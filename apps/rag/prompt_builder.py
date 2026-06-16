"""
Prompt construction for RAG.
Builds a grounded prompt that forces the LLM to cite sources
and refuses to answer from outside the provided context.
"""
from __future__ import annotations
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are DomainGPT, a precise and reliable AI assistant.

Your job is to answer the user's question using ONLY the context passages provided below.

Rules:
1. Base your answer exclusively on the provided context. Do NOT use prior knowledge.
2. If the answer cannot be found in the context, say exactly: "I don't have enough information in the provided documents to answer this question."
3. Always cite the source document and page number at the end of every factual statement using [Source: <filename>, p.<page>] format.
4. Be concise and factual. Do not speculate.
5. If the question asks about multiple topics, address each one separately.

--- CONTEXT START ---
{context}
--- CONTEXT END ---
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query optimizer. Rewrite the user's question to be more specific and "
        "retrieval-friendly. Output ONLY the rewritten query, nothing else.",
    ),
    ("human", "Original question: {question}"),
])


def build_context(docs: list[Document]) -> str:
    """
    Concatenate retrieved chunks into a formatted context string.

    Args:
        docs: Reranked LangChain Document list.

    Returns:
        Formatted context block with per-chunk citations.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        filename = meta.get("filename", "unknown")
        page = meta.get("page_num", "?")
        parts.append(f"[{i}] File: {filename} | Page: {page}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_citation_list(docs: list[Document]) -> list[dict]:
    """
    Build a structured citation list for the API response.

    Returns:
        List of {index, filename, page_num, chunk_index, score}.
    """
    return [
        {
            "index": i + 1,
            "filename": doc.metadata.get("filename", "unknown"),
            "page_num": doc.metadata.get("page_num"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "rerank_score": doc.metadata.get("rerank_score"),
        }
        for i, doc in enumerate(docs)
    ]
