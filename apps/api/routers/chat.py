"""
Chat router – synchronous and streaming RAG-powered chat endpoints.
Persists conversation history in PostgreSQL.
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from apps.api.core.database import get_db
from apps.api.auth.dependencies import get_current_user
from apps.api.models.user import User
from apps.api.models.conversation import Conversation, Message
from apps.rag.pipeline import run_rag, run_rag_streaming
from apps.api.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None   # pass to continue existing conversation


class Citation(BaseModel):
    index: int
    filename: str
    page_num: int | None
    chunk_index: int | None
    rerank_score: float | None


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    citations: list[Citation]
    rewritten_query: str


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_or_create_conversation(
    conversation_id: str | None,
    user: User,
    db: AsyncSession,
) -> Conversation:
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == user.tenant_id,
            )
        )
        conv = result.scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv

    conv = Conversation(
        id=str(uuid.uuid4()),
        tenant_id=user.tenant_id,
        user_id=user.id,
    )
    db.add(conv)
    await db.flush()
    return conv


async def _load_history(conversation_id: str, db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    msgs = result.scalars().all()
    # Return last 10 turns to keep context manageable
    return [{"role": m.role, "content": m.content} for m in msgs[-10:]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Send a question and get a grounded RAG answer with citations.
    Creates a new conversation or continues an existing one.
    """
    conv = await _get_or_create_conversation(body.conversation_id, current_user, db)
    history = await _load_history(conv.id, db)

    # ── Run RAG pipeline ──────────────────────────────────────────────────
    result = await run_rag(
        question=body.question,
        tenant_id=current_user.tenant_id,
        chat_history=history,
    )

    # ── Persist messages ──────────────────────────────────────────────────
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="human",
        content=body.question,
    )
    ai_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        role="assistant",
        content=result["answer"],
        sources=result["citations"],
    )
    db.add(user_msg)
    db.add(ai_msg)
    await db.flush()

    logger.info("chat_response", conv_id=conv.id, citations=len(result["citations"]))
    return ChatResponse(
        conversation_id=conv.id,
        message_id=ai_msg.id,
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        rewritten_query=result["rewritten_query"],
    )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Stream the RAG answer token-by-token using Server-Sent Events.
    Conversation is saved after the stream completes.
    """
    conv = await _get_or_create_conversation(body.conversation_id, current_user, db)
    history = await _load_history(conv.id, db)

    async def token_generator():
        full_response = []
        async for token in run_rag_streaming(
            question=body.question,
            tenant_id=current_user.tenant_id,
            chat_history=history,
        ):
            full_response.append(token)
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

        # Persist after streaming
        user_msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role="human",
            content=body.question,
        )
        ai_msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role="assistant",
            content="".join(full_response),
        )
        db.add(user_msg)
        db.add(ai_msg)

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Conversation-Id": conv.id,
        },
    )


@router.get("/conversations", summary="List user conversations")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return all conversations for the current user."""
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.user_id == current_user.id,
            Conversation.tenant_id == current_user.tenant_id,
        )
        .order_by(Conversation.created_at.desc())
    )
    convs = result.scalars().all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at} for c in convs]
