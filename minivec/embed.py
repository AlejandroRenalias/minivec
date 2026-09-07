"""Turn text into vectors with a local sentence-transformers model.

Stage 1. The model is loaded lazily on first use (the first call downloads
~90 MB, then it runs offline). Output vectors are L2-normalized so that a plain
dot product between two of them equals their cosine similarity.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Wraps a sentence-transformers model behind a tiny interface."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None  # loaded on first use

    def _ensure_model(self):
        if self._model is None:
            # imported lazily so `import minivec` doesn't pull in torch
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors this model produces."""
        return int(self._ensure_model().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of strings.

        Returns:
            float32 array of shape ``(len(texts), dim)``, each row L2-normalized.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        vectors = self._ensure_model().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)
