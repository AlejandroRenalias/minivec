"""Measure the IVF index against brute-force ground truth.

Stage 3. For a set of query vectors:

  * brute force gives the *true* top-k for each query
  * IVF gives an approximate top-k, faster

We report **recall@k** (fraction of the true top-k that IVF also returned) and the
mean query latency of each, swept over a few ``nprobe`` values so the
speed/recall trade-off is visible.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("stage 3")


if __name__ == "__main__":
    main()
