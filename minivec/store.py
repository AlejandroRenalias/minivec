"""Hold vectors plus their metadata, and persist them to disk.

Stage 2. A Store is the vector matrix (row ``i`` is chunk ``i``) alongside a
parallel list of :class:`~minivec.chunk.Chunk`. On disk it is a directory:

    <dir>/vectors.npy    float32 array, shape (n, dim)
    <dir>/meta.jsonl     one JSON object per chunk, in row order
    <dir>/config.json    model name, dim, count
"""

from __future__ import annotations

import json
from dataclasses import asdict
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
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of shape (_, {self.dim}), got {vectors.shape}"
            )
        if len(vectors) != len(chunks):
            raise ValueError(
                f"{len(vectors)} vectors but {len(chunks)} chunks"
            )
        self.vectors = (
            vectors if len(self.vectors) == 0 else np.vstack([self.vectors, vectors])
        )
        self.chunks.extend(chunks)

    def save(self, path: str | Path) -> None:
        """Write this store to a directory (created if needed)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "meta.jsonl", "w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        with open(path / "config.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "dim": self.dim,
                    "model_name": self.model_name,
                    "count": len(self.chunks),
                },
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Store":
        """Read a store previously written by :meth:`save`."""
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        store = cls(dim=int(config["dim"]), model_name=config["model_name"])
        store.vectors = np.load(path / "vectors.npy").astype(np.float32, copy=False)

        chunks: list[Chunk] = []
        with open(path / "meta.jsonl", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(Chunk(**json.loads(line)))
        store.chunks = chunks

        if len(store.vectors) != len(store.chunks):
            raise ValueError(
                f"corrupt store: {len(store.vectors)} vectors vs "
                f"{len(store.chunks)} metadata rows"
            )
        return store

    def __len__(self) -> int:
        return len(self.chunks)
