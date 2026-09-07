"""Measure the IVF index against brute-force ground truth.

Stage 3. Builds a synthetic, mildly-clustered vector set, takes brute force as
the source of truth, then sweeps IVF's ``nprobe`` and reports, for each setting:

  * recall@k  -- fraction of the true top-k that IVF also returned
  * ms/query  -- mean search latency
  * speedup   -- versus brute force

Run:  python benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

from minivec.index import BruteForceIndex, IVFIndex

N_VECTORS = 200_000
DIM = 128
N_QUERIES = 300
K = 10
NLIST = 1024
NPROBE_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
NOISE = 0.9  # blob spread; higher => neighbours leak across clusters => lower recall
SEED = 0


def make_dataset(n: int, dim: int, seed: int) -> np.ndarray:
    """n unit vectors drawn from ~sqrt(n) latent blobs, so clustering helps."""
    rng = np.random.default_rng(seed)
    n_blobs = max(1, int(n**0.5))
    centers = rng.standard_normal((n_blobs, dim))
    which = rng.integers(0, n_blobs, size=n)
    vectors = centers[which] + NOISE * rng.standard_normal((n, dim))
    vectors = vectors.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def mean_latency_ms(index, queries: np.ndarray, k: int) -> float:
    start = time.perf_counter()
    for q in queries:
        index.search(q, k)
    return (time.perf_counter() - start) / len(queries) * 1e3


def recall_at_k(truth_ids: np.ndarray, got_ids: np.ndarray) -> float:
    hits = sum(
        len(set(t.tolist()) & set(g.tolist())) for t, g in zip(truth_ids, got_ids)
    )
    return hits / truth_ids.size


def main() -> None:
    print(
        f"dataset: {N_VECTORS:,} vectors x {DIM}d   queries: {N_QUERIES}   "
        f"k: {K}   nlist: {NLIST}\n"
    )

    vectors = make_dataset(N_VECTORS, DIM, SEED)
    rng = np.random.default_rng(SEED + 1)
    queries = vectors[rng.integers(0, N_VECTORS, size=N_QUERIES)]

    brute = BruteForceIndex()
    brute.build(vectors)
    truth_ids = np.stack([brute.search(q, K)[1] for q in queries])
    brute_ms = mean_latency_ms(brute, queries, K)
    print(f"brute force:  {brute_ms:6.2f} ms/query   (recall 1.00 by definition)\n")

    ivf = IVFIndex(nlist=NLIST, nprobe=1, seed=SEED)
    build_start = time.perf_counter()
    ivf.build(vectors)
    print(f"IVF build (k-means): {time.perf_counter() - build_start:.2f} s\n")

    print(f"{'nprobe':>7} {'recall@k':>10} {'ms/query':>10} {'speedup':>9}")
    print("-" * 40)
    for nprobe in NPROBE_SWEEP:
        ivf.nprobe = nprobe
        got_ids = np.stack([ivf.search(q, K)[1] for q in queries])
        recall = recall_at_k(truth_ids, got_ids)
        ms = mean_latency_ms(ivf, queries, K)
        print(f"{nprobe:>7} {recall:>10.3f} {ms:>10.3f} {brute_ms / ms:>8.1f}x")

    print(
        "\nReading it: more probes -> higher recall, less speedup. The useful "
        "region is where recall is ~0.95+ while still several times faster than "
        "brute force."
    )


if __name__ == "__main__":
    main()
