"""Tests for minivec.chunk — filled in during stage 1."""

import pytest

from minivec.chunk import Chunk, chunk_text


@pytest.mark.skip(reason="stage 1")
def test_chunks_cover_text_with_overlap():
    text = "sentence. " * 200
    chunks = chunk_text(text, source="doc.txt", chunk_size=100, overlap=20)
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].start == 0
    # consecutive chunks should overlap by ~`overlap` chars
    assert chunks[1].start < chunks[0].start + 100
