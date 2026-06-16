"""
Plain-text file parser – handles UTF-8 and latin-1 encodings.
"""
from __future__ import annotations
from pathlib import Path


def extract_txt(source: str | Path | bytes) -> str:
    """
    Return the decoded text from a .txt file.

    Args:
        source: file path, Path, or raw bytes.

    Returns:
        Plain text string.
    """
    if isinstance(source, bytes):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return source.decode(enc)
            except UnicodeDecodeError:
                continue
        return source.decode("utf-8", errors="replace")

    path = Path(source)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")
