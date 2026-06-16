"""
Unit tests for the ingestion pipeline.
"""
import pytest
from apps.ingestion.chunking import chunk_text
from apps.ingestion.metadata import get_file_type, build_metadata
from apps.ingestion.txt_parser import extract_txt


def test_chunk_text_basic():
    text = "Hello world. " * 100
    chunks = chunk_text(text, metadata={"filename": "test.txt"})
    assert len(chunks) > 1
    for chunk in chunks:
        assert "content" in chunk
        assert "chunk_index" in chunk
        assert len(chunk["content"]) <= 600  # chunk_size + some overlap tolerance


def test_chunk_text_preserves_metadata():
    chunks = chunk_text("Some text", metadata={"doc_id": "abc123"})
    for chunk in chunks:
        assert chunk["metadata"]["doc_id"] == "abc123"


def test_get_file_type():
    assert get_file_type("report.pdf") == "pdf"
    assert get_file_type("notes.docx") == "docx"
    assert get_file_type("readme.txt") == "txt"
    assert get_file_type("photo.png") == "image"
    assert get_file_type("unknown.xyz") == "unknown"


def test_build_metadata():
    content = b"hello world"
    meta = build_metadata("test.pdf", content, "tenant1", "user1")
    assert meta["filename"] == "test.pdf"
    assert meta["file_type"] == "pdf"
    assert meta["sha256"] is not None
    assert meta["tenant_id"] == "tenant1"


def test_extract_txt_utf8():
    text = b"Hello, this is a test document."
    result = extract_txt(text)
    assert "Hello" in result


def test_extract_txt_latin1():
    text = "caf\xe9".encode("latin-1")
    result = extract_txt(text)
    assert result is not None
