"""End-to-end tests for minivec.db.

Need `sentence-transformers` (real embeddings); skipped automatically if missing.
"""

import pytest

pytest.importorskip("sentence_transformers")

from minivec.db import MiniVec  # noqa: E402


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    (d / "auth.md").write_text(
        "The login flow uses short-lived OAuth tokens refreshed every hour. "
        "Passwords were dropped in the March redesign."
    )
    (d / "kitchen.md").write_text(
        "The kitchen was repainted a pale shade of green over the weekend. "
        "New tiles go in next month."
    )
    (d / "notes.txt").write_text(
        "Remember to buy milk, eggs, and coffee on the way home."
    )
    return d


def test_ingest_then_query_finds_the_relevant_file(corpus):
    db = MiniVec()
    n = db.ingest(corpus)
    assert n >= 3

    results = db.query("how does signing in work?", k=1)
    assert results[0].chunk.source.endswith("auth.md")
    assert 0.0 <= results[0].score <= 1.0


def test_query_results_are_ranked(corpus):
    db = MiniVec()
    db.ingest(corpus)
    results = db.query("painting the house", k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_save_load_roundtrip_preserves_search(tmp_path, corpus):
    db = MiniVec()
    db.ingest(corpus)
    before = db.query("oauth login tokens", k=1)[0].chunk.source
    db.save(tmp_path / "store")

    reloaded = MiniVec.load(tmp_path / "store")
    after = reloaded.query("oauth login tokens", k=1)[0].chunk.source
    assert before == after


def test_query_without_ingest_raises():
    with pytest.raises(RuntimeError):
        MiniVec().query("anything", k=1)
