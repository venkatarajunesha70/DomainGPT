"""
LangChain RAG pipeline with LangGraph orchestration.
Full flow: query rewrite → hybrid retrieval → reranking → grounded answer.
Supports streaming and non-streaming modes.
"""
from __future__ import annotations
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from apps.rag.retriever import retrieve
from apps.rag.hybrid_search import hybrid_retrieve
from apps.rag.reranker import rerank
from apps.rag.prompt_builder import (
    CHAT_PROMPT,
    QUERY_REWRITE_PROMPT,
    build_context,
    build_citation_list,
)
from apps.api.core.config import get_settings
from apps.api.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── LLM client ───────────────────────────────────────────────────────────────

def _get_llm(streaming: bool = False) -> ChatGroq:
    """Return a Groq ChatGroq client (wraps open-weight models like Llama 3)."""
    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0.1,
        streaming=streaming,
    )


# ── LangGraph state ───────────────────────────────────────────────────────────

class RAGState(dict):
    """Typed state dict used by LangGraph nodes."""
    question: str
    rewritten_query: str
    tenant_id: str
    chat_history: list
    docs: list
    context: str
    answer: str
    citations: list


# ── Graph nodes ───────────────────────────────────────────────────────────────

async def rewrite_query_node(state: RAGState) -> RAGState:
    """Rewrite the user question for better retrieval."""
    llm = _get_llm()
    chain = QUERY_REWRITE_PROMPT | llm
    result = await chain.ainvoke({"question": state["question"]})
    rewritten = result.content.strip()
    logger.debug("query_rewritten", original=state["question"], rewritten=rewritten)
    return {**state, "rewritten_query": rewritten}


async def retrieve_node(state: RAGState) -> RAGState:
    """Vector + hybrid retrieval from Pinecone."""
    query = state.get("rewritten_query") or state["question"]
    dense_docs = retrieve(query=query, tenant_id=state["tenant_id"], top_k=10)
    fused_docs = hybrid_retrieve(query=query, dense_docs=dense_docs, top_k=8)
    return {**state, "docs": fused_docs}


async def rerank_node(state: RAGState) -> RAGState:
    """Cross-encoder reranking to select best 4 chunks."""
    query = state.get("rewritten_query") or state["question"]
    reranked = rerank(query=query, docs=state["docs"], top_n=4)
    context = build_context(reranked)
    return {**state, "docs": reranked, "context": context}


async def generate_node(state: RAGState) -> RAGState:
    """Generate the final grounded answer."""
    llm = _get_llm()
    chain = CHAT_PROMPT | llm

    # Convert stored history to LangChain message objects
    history = []
    for msg in state.get("chat_history", []):
        if msg["role"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    result = await chain.ainvoke({
        "context": state["context"],
        "chat_history": history,
        "question": state["question"],
    })
    citations = build_citation_list(state["docs"])
    return {**state, "answer": result.content, "citations": citations}


# ── Build the LangGraph DAG ───────────────────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


_rag_graph = None


def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


# ── Public API ────────────────────────────────────────────────────────────────

async def run_rag(
    question: str,
    tenant_id: str,
    chat_history: list | None = None,
) -> dict:
    """
    Run the full RAG pipeline and return answer + citations.

    Args:
        question:     User's question.
        tenant_id:    Tenant identifier for document isolation.
        chat_history: List of {role, content} dicts for conversation memory.

    Returns:
        Dict with keys: answer, citations, rewritten_query.
    """
    graph = get_rag_graph()
    initial_state: RAGState = {
        "question": question,
        "rewritten_query": "",
        "tenant_id": tenant_id,
        "chat_history": chat_history or [],
        "docs": [],
        "context": "",
        "answer": "",
        "citations": [],
    }
    result = await graph.ainvoke(initial_state)
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "rewritten_query": result["rewritten_query"],
    }


async def run_rag_streaming(
    question: str,
    tenant_id: str,
    chat_history: list | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream the RAG answer token-by-token.
    Retrieval and reranking happen synchronously; only generation is streamed.

    Yields:
        Token strings from the LLM.
    """
    # Run retrieval steps
    rewrite_llm = _get_llm()
    rewrite_chain = QUERY_REWRITE_PROMPT | rewrite_llm
    rewrite_result = await rewrite_chain.ainvoke({"question": question})
    rewritten = rewrite_result.content.strip()

    dense_docs = retrieve(query=rewritten, tenant_id=tenant_id, top_k=10)
    fused_docs = hybrid_retrieve(query=rewritten, dense_docs=dense_docs, top_k=8)
    reranked = rerank(query=rewritten, docs=fused_docs, top_n=4)
    context = build_context(reranked)

    history = []
    for msg in (chat_history or []):
        if msg["role"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    streaming_llm = _get_llm(streaming=True)
    chain = CHAT_PROMPT | streaming_llm

    async for chunk in chain.astream({
        "context": context,
        "chat_history": history,
        "question": question,
    }):
        if chunk.content:
            yield chunk.content
