"""Tests for minivec.index — filled in during stages 2 and 3."""

import numpy as np
import pytest

from minivec.index import BruteForceIndex, IVFIndex


def _random_normalized(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.mark.skip(reason="stage 2")
def test_brute_force_finds_the_exact_nearest():
    vectors = _random_normalized(1000, 32)
    idx = BruteForceIndex()
    idx.build(vectors)
    scores, ids = idx.search(vectors[42], k=1)
    assert ids[0] == 42
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skip(reason="stage 3")
def test_ivf_recall_is_high_against_brute_force():
    vectors = _random_normalized(5000, 32)
    brute = BruteForceIndex(); brute.build(vectors)
    ivf = IVFIndex(nlist=64, nprobe=16); ivf.build(vectors)

    hits = total = 0
    for q in vectors[:100]:
        _, truth = brute.search(q, k=10)
        _, approx = ivf.search(q, k=10)
        hits += len(set(truth) & set(approx))
        total += 10
    assert hits / total > 0.8
