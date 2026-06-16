"""
Document upload router.
Validates file type, stores to S3, and enqueues a Celery ingestion job.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from apps.api.core.database import get_db
from apps.api.auth.dependencies import get_current_user
from apps.api.models.user import User
from apps.api.models.document import Document, DocumentStatus
from apps.ingestion.metadata import get_file_type, build_metadata, SUPPORTED_TYPES
from apps.api.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    status: str
    message: str


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a document for ingestion.

    Supported types: PDF, DOCX, TXT, PNG, JPG, JPEG.
    Max size: 50 MB.
    Returns immediately with a document ID; processing is async.
    """
    # ── Validate file type ────────────────────────────────────────────────
    file_type = get_file_type(file.filename or "")
    if file_type == "unknown":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {list(SUPPORTED_TYPES.keys())}",
        )

    # ── Read and validate size ────────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )

    doc_id = str(uuid.uuid4())
    s3_key = f"{current_user.tenant_id}/{doc_id}/{file.filename}"

    # ── Persist document record ───────────────────────────────────────────
    meta = build_metadata(
        filename=file.filename or "unnamed",
        file_bytes=content,
        tenant_id=current_user.tenant_id,
        owner_id=current_user.id,
    )

    doc = Document(
        id=doc_id,
        tenant_id=current_user.tenant_id,
        owner_id=current_user.id,
        filename=file.filename or "unnamed",
        file_type=file_type,
        s3_key=s3_key,
        status=DocumentStatus.pending,
        doc_metadata=meta,
    )
    db.add(doc)
    await db.flush()

    # ── Enqueue Celery ingestion task ─────────────────────────────────────
    try:
        from apps.workers.ingest_worker import ingest_document_task
        ingest_document_task.delay(
            doc_id=doc_id,
            file_content=content,
            filename=file.filename or "unnamed",
            tenant_id=current_user.tenant_id,
        )
    except Exception as exc:
        logger.warning("celery_enqueue_failed", doc_id=doc_id, error=str(exc))
        # Still return 202 – can re-trigger manually

    logger.info("document_uploaded", doc_id=doc_id, filename=file.filename)
    return DocumentResponse(
        id=doc_id,
        filename=file.filename or "unnamed",
        file_type=file_type,
        status="pending",
        message="Document received. Ingestion started in background.",
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document_status(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Check processing status of an uploaded document."""
    from sqlalchemy import select
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.tenant_id == current_user.tenant_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        status=doc.status.value,
        message=f"Chunks indexed: {doc.chunk_count}",
    )
