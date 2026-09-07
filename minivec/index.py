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


class BruteForceIndex:
    """Exact nearest-neighbour search by full dot-product scan."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("stage 2")

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, ids)`` for the top ``k`` rows, best first."""
        raise NotImplementedError("stage 2")


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
