"""
PDF parser – extracts text and preserves page-level metadata.
Uses pdfplumber for digital PDFs; falls back to Tesseract OCR for scanned pages.
"""
from __future__ import annotations
import io
from pathlib import Path
from typing import Generator

import pdfplumber
from PIL import Image

from apps.ingestion.image_ocr import ocr_image
from apps.api.core.logging import get_logger

logger = get_logger(__name__)


def extract_pdf(source: str | Path | bytes) -> Generator[dict, None, None]:
    """
    Yield page dicts: {page_num, text, has_tables, is_ocr, source}.

    Args:
        source: file path, Path object, or raw bytes of the PDF.

    Yields:
        dict with keys: page_num, text, has_tables, is_ocr, source.
    """
    if isinstance(source, bytes):
        data = io.BytesIO(source)
        src_name = "<bytes>"
    else:
        data = str(source)
        src_name = str(source)

    with pdfplumber.open(data) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            has_tables = len(page.extract_tables()) > 0
            is_ocr = False

            # If page yields very little text, try OCR on the page image
            if len(text.strip()) < 50:
                try:
                    img: Image.Image = page.to_image(resolution=300).original
                    text = ocr_image(img)
                    is_ocr = True
                    logger.debug("ocr_fallback", page=i, source=src_name)
                except Exception as exc:
                    logger.warning("ocr_failed", page=i, error=str(exc))

            yield {
                "page_num": i,
                "text": text,
                "has_tables": has_tables,
                "is_ocr": is_ocr,
                "source": src_name,
            }
