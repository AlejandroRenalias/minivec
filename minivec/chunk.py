"""Split raw text into overlapping chunks.

Stage 1. An embedding model can only look at a bounded amount of text at once, and
smaller chunks give more precise retrieval. We slide a fixed-size window over the
text with a little overlap so a sentence spanning a boundary isn't lost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One unit of indexed text.

    Attributes:
        text:   the chunk's content
        source: where it came from (e.g. a file path)
        start:  character offset of this chunk within its source
    """

    text: str
    source: str
    start: int


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Break ``text`` into overlapping :class:`Chunk`s.

    A window of ``chunk_size`` characters slides across ``text`` advancing by
    ``chunk_size - overlap`` each step, so consecutive chunks share ``overlap``
    characters. Whitespace-only windows are dropped; the final chunk may be
    shorter than ``chunk_size``.

    Args:
        text:       the full document text
        source:     identifier stored on every produced chunk
        chunk_size: window length in characters (must be > 0)
        overlap:    characters shared between consecutive chunks; must satisfy
                    ``0 <= overlap < chunk_size``

    Returns:
        chunks in document order
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(text), step):
        window = text[start : start + chunk_size]
        if window.strip():
            chunks.append(Chunk(text=window, source=source, start=start))
        if start + chunk_size >= len(text):
            break
    return chunks
