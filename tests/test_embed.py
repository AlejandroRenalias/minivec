"""Tests for minivec.embed.

These need `sentence-transformers` installed; the first run also downloads the
~90 MB model. They are skipped automatically if the package is missing.
"""

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from minivec.embed import Embedder  # noqa: E402


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_embed_shape_and_normalization(embedder):
    vecs = embedder.embed(["hello world", "a completely unrelated sentence"])

    assert vecs.shape == (2, embedder.dim)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_similar_text_scores_higher_than_unrelated(embedder):
    v = embedder.embed(
        [
            "How do I reset my password?",
            "I forgot my login credentials.",
            "The recipe calls for two cups of flour.",
        ]
    )
    related = float(v[0] @ v[1])
    unrelated = float(v[0] @ v[2])
    assert related > unrelated


def test_embed_empty_list_returns_empty_matrix(embedder):
    vecs = embedder.embed([])
    assert vecs.shape == (0, embedder.dim)
