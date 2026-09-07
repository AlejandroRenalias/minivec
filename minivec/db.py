"""The top-level object that ties everything together.

Stage 2. ``MiniVec`` owns an :class:`~minivec.embed.Embedder`, a
:class:`~minivec.store.Store`, and an index. ``ingest`` walks a path, chunks and
embeds every text file, and adds the result to the store. ``query`` embeds a
question and asks the index for the closest chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunk import Chunk
from .embed import DEFAULT_MODEL, Embedder
from .store import Store

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class MiniVec:
    def __init__(self, model_name: str = DEFAULT_MODEL, index: str = "brute") -> None:
        self.embedder = Embedder(model_name)
        self.index_kind = index
        self.store: Store | None = None
        self._index = None

    # -- building -----------------------------------------------------------
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
        raise NotImplementedError("stage 2")

    # -- querying ---------------------------------------------------------------
    def query(self, text: str, k: int = 5) -> list[SearchResult]:
        """Return the ``k`` stored chunks closest in meaning to ``text``."""
        raise NotImplementedError("stage 2")

    # -- persistence ------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        raise NotImplementedError("stage 2")

    @classmethod
    def load(cls, path: str | Path, index: str = "brute") -> "MiniVec":
        raise NotImplementedError("stage 2")
