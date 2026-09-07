"""Search strategies over a matrix of normalized vectors.

Every index implements the same interface:

    build(vectors)            -> None
    search(query, k)          -> (scores, ids)   best first

- ``BruteForceIndex`` (stage 2): compare the query against every vector. Exact,
  O(n) per query — the ground truth the IVF index is measured against.
- ``IVFIndex`` (stage 3): k-means the vectors into ``nlist`` clusters, then only
  compare against the ``nprobe`` nearest clusters. Fewer comparisons, approximate.
"""

from __future__ import annotations

import numpy as np


def _top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and scores of the ``k`` largest entries, highest first."""
    n = len(scores)
    if n == 0 or k <= 0:
        empty_i = np.empty(0, dtype=np.int64)
        return scores[empty_i], empty_i
    k = min(k, n)
    # argpartition gets the k best in O(n); then sort just those k
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="stable")]
    return scores[order], order


class BruteForceIndex:
    """Exact nearest-neighbour search by full dot-product scan."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, ids)`` for the top ``k`` rows, best first."""
        if self._vectors is None:
            raise RuntimeError("index has not been built")
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} != index dim {self._vectors.shape[1]}"
            )
        scores = self._vectors @ q
        return _top_k(scores, k)


class IVFIndex:
    """Approximate search via k-means clustering (inverted file index)."""

    def __init__(self, nlist: int = 64, nprobe: int = 8, seed: int = 0) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self._centroids: np.ndarray | None = None
        self._postings: list[np.ndarray] | None = None  # row ids per cluster
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("stage 3")

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("stage 3")
