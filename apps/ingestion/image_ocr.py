"""
OCR module – wraps Tesseract (pytesseract) with pre-processing for better accuracy.
Also supports vision LLM as a high-quality fallback.
"""
from __future__ import annotations
import io
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

from apps.api.core.logging import get_logger

logger = get_logger(__name__)


def _preprocess(img: Image.Image) -> Image.Image:
    """Improve OCR accuracy: grayscale → contrast → sharpen."""
    img = img.convert("L")                                  # grayscale
    img = ImageEnhance.Contrast(img).enhance(2.0)           # boost contrast
    img = img.filter(ImageFilter.SHARPEN)                   # sharpen edges
    return img


def ocr_image(source: str | Path | bytes | Image.Image) -> str:
    """
    Run Tesseract OCR on an image and return the extracted text.

    Args:
        source: PIL Image, file path, Path, or raw image bytes.

    Returns:
        Extracted text string.
    """
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(str(source))

    processed = _preprocess(img)

    config = "--oem 3 --psm 6"   # LSTM engine, assume uniform block of text
    text: str = pytesseract.image_to_string(processed, config=config)
    logger.debug("ocr_complete", chars=len(text))
    return text.strip()
