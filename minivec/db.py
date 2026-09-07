"""The top-level object that ties everything together.

Stage 2. ``MiniVec`` owns an :class:`~minivec.embed.Embedder`, a
:class:`~minivec.store.Store`, and an index. ``ingest`` walks a path, chunks and
embeds every text file, and adds the result to the store. ``query`` embeds a
question and asks the index for the closest chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunk import Chunk, chunk_text
from .embed import DEFAULT_MODEL, Embedder
from .index import BruteForceIndex, IVFIndex
from .store import Store

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def _make_index(kind: str):
    if kind == "brute":
        return BruteForceIndex()
    if kind == "ivf":
        return IVFIndex()
    raise ValueError(f"unknown index kind: {kind!r} (expected 'brute' or 'ivf')")


class MiniVec:
    def __init__(self, model_name: str = DEFAULT_MODEL, index: str = "brute") -> None:
        self.embedder = Embedder(model_name)
        self.index_kind = index
        self.store: Store | None = None
        self._index = None  # built lazily, invalidated on ingest

    # -- building ------------------------------------------------------------
    def _iter_text_files(self, path: Path):
        if path.is_file():
            yield path
            return
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
                yield p

    def ingest(
        self,
        path: str | Path,
        *,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> int:
        """Chunk, embed and store every text file under ``path``.

        Returns the number of chunks added.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        chunks: list[Chunk] = []
        for file in self._iter_text_files(path):
            text = file.read_text(encoding="utf-8", errors="replace")
            chunks.extend(
                chunk_text(
                    text,
                    source=str(file),
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
        if not chunks:
            return 0

        vectors = self.embedder.embed([c.text for c in chunks])
        if self.store is None:
            self.store = Store(
                dim=vectors.shape[1], model_name=self.embedder.model_name
            )
        self.store.add(vectors, chunks)
        self._index = None
        return len(chunks)

    # -- querying ------------------------------------------------------------
    def _ensure_index(self):
        if self.store is None or len(self.store) == 0:
            raise RuntimeError("nothing has been ingested yet")
        if self._index is None:
            self._index = _make_index(self.index_kind)
            self._index.build(self.store.vectors)
        return self._index

    def query(self, text: str, k: int = 5) -> list[SearchResult]:
        """Return the ``k`` stored chunks closest in meaning to ``text``."""
        index = self._ensure_index()
        query_vec = self.embedder.embed([text])[0]
        scores, ids = index.search(query_vec, k)
        assert self.store is not None  # for type checkers; _ensure_index guarantees it
        return [
            SearchResult(chunk=self.store.chunks[int(i)], score=float(s))
            for s, i in zip(scores, ids)
        ]

    # -- persistence ------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.store is None:
            raise RuntimeError("nothing to save")
        self.store.save(path)

    @classmethod
    def load(cls, path: str | Path, index: str = "brute") -> "MiniVec":
        store = Store.load(path)
        db = cls(model_name=store.model_name, index=index)
        db.store = store
        return db
