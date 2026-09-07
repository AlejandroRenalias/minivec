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


def _kmeans(
    vectors: np.ndarray, k: int, *, seed: int, max_iter: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means on unit vectors.

    Similarity is the dot product (the vectors are L2-normalized), so a point
    belongs to the centroid it points most towards, and each centroid is the
    renormalized mean of its members.

    Returns ``(centroids, labels)`` with shapes ``(k, dim)`` and ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    n = len(vectors)
    # init: k distinct rows chosen at random
    centroids = vectors[rng.choice(n, size=k, replace=False)].copy()

    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(max_iter):
        new_labels = (vectors @ centroids.T).argmax(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = vectors[labels == c]
            if len(members) == 0:
                # empty cluster: reseed it on a random point
                centroids[c] = vectors[rng.integers(n)]
                continue
            centroids[c] = members.mean(axis=0)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids /= np.where(norms == 0, 1.0, norms)
    return centroids, labels


class IVFIndex:
    """Approximate search via k-means clustering (inverted file index).

    ``build`` partitions the vectors into ``nlist`` clusters. ``search`` scores
    the query against the cluster centroids, then brute-forces only the vectors
    in the ``nprobe`` nearest clusters. Fewer comparisons, so faster — at the
    cost of missing a true neighbour that happens to sit in an unprobed cluster.
    """

    def __init__(self, nlist: int = 64, nprobe: int = 8, seed: int = 0) -> None:
        if nlist < 1 or nprobe < 1:
            raise ValueError("nlist and nprobe must be >= 1")
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self._centroids: np.ndarray | None = None
        self._postings: list[np.ndarray] | None = None  # row ids per cluster
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(self._vectors)
        k = max(1, min(self.nlist, n))

        centroids, labels = _kmeans(self._vectors, k, seed=self.seed)
        self._centroids = centroids
        self._postings = [
            np.nonzero(labels == c)[0].astype(np.int64) for c in range(k)
        ]

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None or self._centroids is None:
            raise RuntimeError("index has not been built")
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} != index dim {self._vectors.shape[1]}"
            )

        # 1. pick the nprobe nearest clusters
        centroid_scores = self._centroids @ q
        n_clusters = len(self._centroids)
        probe = min(self.nprobe, n_clusters)
        probe_clusters = np.argpartition(-centroid_scores, probe - 1)[:probe]

        # 2. gather their members and brute-force just those
        assert self._postings is not None
        candidates = np.concatenate([self._postings[c] for c in probe_clusters])
        if len(candidates) == 0:
            empty = np.empty(0, dtype=np.int64)
            return self._vectors[empty] @ q, empty

        cand_scores = self._vectors[candidates] @ q
        local_scores, local_ids = _top_k(cand_scores, k)
        return local_scores, candidates[local_ids]
