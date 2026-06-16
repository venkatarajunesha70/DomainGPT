"""
Celery ingestion worker.
Full pipeline: parse → chunk → embed → upsert to Pinecone → update DB.
"""
from __future__ import annotations
import uuid
from apps.workers.celery_app import celery_app
from apps.api.core.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="ingest_document",
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
)
def ingest_document_task(
    self,
    doc_id: str,
    file_content: bytes,
    filename: str,
    tenant_id: str,
) -> dict:
    """
    Full document ingestion pipeline (runs in Celery worker process).

    1. Detect file type and parse text.
    2. Semantic chunking.
    3. Embed chunks.
    4. Upsert to Pinecone.
    5. Update document status in PostgreSQL.

    Args:
        doc_id:       Database document ID.
        file_content: Raw file bytes.
        filename:     Original filename.
        tenant_id:    Tenant namespace.

    Returns:
        Dict with chunk_count and status.
    """
    from apps.ingestion.metadata import get_file_type
    from apps.ingestion.chunking import chunk_text
    from apps.ingestion.embeddings import embed_texts
    from apps.rag.retriever import upsert_chunks

    logger.info("ingest_start", doc_id=doc_id, filename=filename)

    # ── 1. Parse ──────────────────────────────────────────────────────────
    file_type = get_file_type(filename)
    text = _parse(file_content, filename, file_type)

    if not text.strip():
        _update_db_status(doc_id, "failed", 0)
        return {"status": "failed", "reason": "empty_text"}

    # ── 2. Chunk ──────────────────────────────────────────────────────────
    base_meta = {
        "filename": filename,
        "document_id": doc_id,
        "tenant_id": tenant_id,
        "file_type": file_type,
    }
    chunks = chunk_text(text, metadata=base_meta)

    # ── 3. Embed ──────────────────────────────────────────────────────────
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)

    # ── 4. Build vector records ───────────────────────────────────────────
    vectors = [
        {
            "chunk_id": f"{doc_id}_{c['chunk_index']}",
            "embedding": embeddings[i],
            "content": c["content"],
            "metadata": c["metadata"],
        }
        for i, c in enumerate(chunks)
    ]

    # ── 5. Upsert to Pinecone ─────────────────────────────────────────────
    upsert_chunks(vectors, tenant_id=tenant_id)

    # ── 6. Update DB ──────────────────────────────────────────────────────
    _update_db_status(doc_id, "indexed", len(chunks))

    logger.info("ingest_complete", doc_id=doc_id, chunks=len(chunks))
    return {"status": "indexed", "chunk_count": len(chunks)}


def _parse(content: bytes, filename: str, file_type: str) -> str:
    """Dispatch to the correct parser based on file type."""
    if file_type == "pdf":
        from apps.ingestion.pdf_parser import extract_pdf
        pages = list(extract_pdf(content))
        return "\n\n".join(p["text"] for p in pages)
    elif file_type == "docx":
        from apps.ingestion.doc_parser import extract_docx
        return extract_docx(content)
    elif file_type == "txt":
        from apps.ingestion.txt_parser import extract_txt
        return extract_txt(content)
    elif file_type == "image":
        from apps.ingestion.image_ocr import ocr_image
        return ocr_image(content)
    return ""


def _update_db_status(doc_id: str, status: str, chunk_count: int) -> None:
    """Synchronous DB update (Celery workers run sync)."""
    from sqlalchemy import create_engine, update
    from apps.api.core.config import get_settings
    from apps.api.models.document import Document, DocumentStatus

    settings = get_settings()
    engine = create_engine(settings.sync_database_url)
    with engine.begin() as conn:
        conn.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(
                status=DocumentStatus[status],
                chunk_count=chunk_count,
            )
        )
