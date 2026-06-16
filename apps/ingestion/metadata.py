"""
Metadata extraction utilities.
Produces a normalised metadata dict for every ingested document.
"""
from __future__ import annotations
import hashlib
from pathlib import Path


SUPPORTED_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "txt",
    ".md": "txt",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".bmp": "image",
}


def get_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return SUPPORTED_TYPES.get(ext, "unknown")


def build_metadata(
    filename: str,
    file_bytes: bytes,
    tenant_id: str,
    owner_id: str,
    extra: dict | None = None,
) -> dict:
    """
    Build a metadata dict for a document.

    Args:
        filename:   Original filename.
        file_bytes: Raw file content (used to compute SHA-256 hash).
        tenant_id:  Tenant owning the document.
        owner_id:   User who uploaded the document.
        extra:      Additional metadata fields.

    Returns:
        Normalised metadata dict.
    """
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    file_type = get_file_type(filename)

    meta = {
        "filename": filename,
        "file_type": file_type,
        "file_size_bytes": len(file_bytes),
        "sha256": sha256,
        "tenant_id": tenant_id,
        "owner_id": owner_id,
    }
    if extra:
        meta.update(extra)
    return meta
