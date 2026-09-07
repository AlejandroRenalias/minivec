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

    Args:
        text:       the full document text
        source:     identifier stored on every produced chunk
        chunk_size: target chunk length in characters
        overlap:    characters shared between consecutive chunks

    Returns:
        chunks in document order
    """
    raise NotImplementedError("stage 1")
