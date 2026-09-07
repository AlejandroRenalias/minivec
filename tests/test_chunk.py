"""Tests for minivec.chunk."""

import pytest

from minivec.chunk import Chunk, chunk_text


def test_chunks_cover_text_with_overlap():
    text = "abcde" * 100  # 500 chars
    chunks = chunk_text(text, source="doc.txt", chunk_size=100, overlap=20)

    # windows advance by 80; the one at 400 already reaches the end, so the
    # loop stops there rather than emitting a redundant trailing sub-slice
    assert [c.start for c in chunks] == [0, 80, 160, 240, 320, 400]
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "doc.txt" for c in chunks)

    # first window is exactly chunk_size, later ones overlap the previous by 20
    assert chunks[0].text == text[0:100]
    assert chunks[1].text[:20] == text[80:100]

    # every character of the source lands in some chunk
    assert chunks[-1].text.endswith(text[-1])
    assert chunks[-1].start + len(chunks[-1].text) == len(text)


def test_whitespace_only_text_yields_no_chunks():
    assert chunk_text("   \n\t  ", source="x") == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("hello world", source="x", chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == Chunk(text="hello world", source="x", start=0)


@pytest.mark.parametrize("overlap", [-1, 10, 20])
def test_bad_overlap_rejected(overlap):
    with pytest.raises(ValueError):
        chunk_text("hello", source="x", chunk_size=10, overlap=overlap)
