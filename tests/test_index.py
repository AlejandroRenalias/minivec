"""Tests for minivec.index."""

import numpy as np
import pytest

from minivec.index import BruteForceIndex, IVFIndex


def _random_normalized(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_brute_force_finds_the_exact_nearest():
    vectors = _random_normalized(1000, 32)
    idx = BruteForceIndex()
    idx.build(vectors)

    scores, ids = idx.search(vectors[42], k=1)
    assert ids[0] == 42
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_brute_force_returns_k_results_sorted_descending():
    vectors = _random_normalized(200, 16)
    idx = BruteForceIndex()
    idx.build(vectors)

    scores, ids = idx.search(vectors[0], k=5)
    assert len(ids) == len(scores) == 5
    assert ids[0] == 0
    assert list(scores) == sorted(scores, reverse=True)


def test_brute_force_k_larger_than_corpus_is_clamped():
    vectors = _random_normalized(3, 8)
    idx = BruteForceIndex()
    idx.build(vectors)

    scores, ids = idx.search(vectors[1], k=99)
    assert len(ids) == 3
    assert set(ids) == {0, 1, 2}


def test_brute_force_rejects_wrong_query_dim():
    idx = BruteForceIndex()
    idx.build(_random_normalized(10, 16))
    with pytest.raises(ValueError):
        idx.search(np.zeros(8, dtype=np.float32), k=1)


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        BruteForceIndex().search(np.zeros(4, dtype=np.float32), k=1)


@pytest.mark.skip(reason="stage 3")
def test_ivf_recall_is_high_against_brute_force():
    vectors = _random_normalized(5000, 32)
    brute = BruteForceIndex()
    brute.build(vectors)
    ivf = IVFIndex(nlist=64, nprobe=16)
    ivf.build(vectors)

    hits = total = 0
    for q in vectors[:100]:
        _, truth = brute.search(q, k=10)
        _, approx = ivf.search(q, k=10)
        hits += len(set(truth.tolist()) & set(approx.tolist()))
        total += 10
    assert hits / total > 0.8
