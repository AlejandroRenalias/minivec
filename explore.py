"""A tiny playground for the one idea minivec rests on:

    similar meaning  ->  similar numbers

Run it:   .venv\\Scripts\\python.exe explore.py

Then edit the SENTENCES list at the bottom, guess the scores before you run it
again, and see if you were right.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def fingerprint(text: str) -> np.ndarray:
    """Turn one string into its 384-number 'meaning fingerprint'."""
    global _model
    if _model is None:
        print("(loading the embedding model, one moment...)\n")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model.encode([text], normalize_embeddings=True)[0]


def similarity(a: str, b: str) -> float:
    """How close in meaning are two strings? 1.0 = identical, 0.0 = unrelated.

    Because every fingerprint is scaled to length 1, this dot product is just
    'how much do these two point in the same direction'.
    """
    return float(np.dot(fingerprint(a), fingerprint(b)))


def show(a: str, b: str) -> None:
    print(f"  {similarity(a, b):+.2f}   {a!r}\n         vs {b!r}\n")


# ---------------------------------------------------------------------------
# EDIT BELOW THIS LINE
# ---------------------------------------------------------------------------

PAIRS = [
    # (sentence A, sentence B)
    ("a dog barked loudly", "a cat sat quietly on the mat"),
    ("a dog barked loudly", "I paid my electricity bill"),
    ("how do I reset my password?", "I forgot my login credentials"),
    ("how do I reset my password?", "what time does the shop close?"),
]

if __name__ == "__main__":
    print("similar meaning -> score near 1.0 | unrelated -> score near 0.0\n")
    for a, b in PAIRS:
        show(a, b)
