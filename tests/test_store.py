"""Tests for minivec.store (no embedding model needed — fake vectors)."""

import numpy as np
import pytest

from minivec.chunk import Chunk
from minivec.store import Store


def _vecs(n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_add_appends_vectors_and_chunks():
    store = Store(dim=4, model_name="fake")
    store.add(_vecs(2, 4), [Chunk("a", "s", 0), Chunk("b", "s", 4)])
    store.add(_vecs(1, 4), [Chunk("c", "s", 8)])

    assert len(store) == 3
    assert store.vectors.shape == (3, 4)
    assert [c.text for c in store.chunks] == ["a", "b", "c"]


def test_add_rejects_shape_and_count_mismatch():
    store = Store(dim=4, model_name="fake")
    with pytest.raises(ValueError):
        store.add(_vecs(2, 8), [Chunk("a", "s", 0), Chunk("b", "s", 1)])
    with pytest.raises(ValueError):
        store.add(_vecs(2, 4), [Chunk("a", "s", 0)])


def test_save_then_load_roundtrips(tmp_path):
    store = Store(dim=4, model_name="fake-model")
    store.add(_vecs(5, 4), [Chunk(f"chunk {i}", "doc.md", i * 4) for i in range(5)])
    store.save(tmp_path / "s")

    loaded = Store.load(tmp_path / "s")
    assert loaded.dim == 4
    assert loaded.model_name == "fake-model"
    assert len(loaded) == 5
    assert loaded.chunks == store.chunks
    np.testing.assert_allclose(loaded.vectors, store.vectors)


def test_load_detects_corruption(tmp_path):
    store = Store(dim=4, model_name="fake")
    store.add(_vecs(3, 4), [Chunk("a", "s", 0), Chunk("b", "s", 1), Chunk("c", "s", 2)])
    store.save(tmp_path / "s")
    # drop a metadata row
    meta = (tmp_path / "s" / "meta.jsonl").read_text().splitlines()
    (tmp_path / "s" / "meta.jsonl").write_text("\n".join(meta[:2]) + "\n")

    with pytest.raises(ValueError):
        Store.load(tmp_path / "s")
