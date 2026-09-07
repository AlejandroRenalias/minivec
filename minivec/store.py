"""Hold vectors plus their metadata, and persist them to disk.

Stage 2. A Store is the vector matrix (row ``i`` is chunk ``i``) alongside a
parallel list of :class:`~minivec.chunk.Chunk`. On disk it is a directory:

    <dir>/vectors.npy    float32 array, shape (n, dim)
    <dir>/meta.jsonl     one JSON object per chunk, in row order
    <dir>/config.json    model name, dim, index type
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .chunk import Chunk


class Store:
    """In-memory collection of vectors + chunks with load/save helpers."""

    def __init__(self, dim: int, model_name: str) -> None:
        self.dim = dim
        self.model_name = model_name
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Append a batch of rows and their matching chunks."""
        raise NotImplementedError("stage 2")

    def save(self, path: str | Path) -> None:
        """Write this store to a directory (created if needed)."""
        raise NotImplementedError("stage 2")

    @classmethod
    def load(cls, path: str | Path) -> "Store":
        """Read a store previously written by :meth:`save`."""
        raise NotImplementedError("stage 2")

    def __len__(self) -> int:
        return len(self.chunks)
