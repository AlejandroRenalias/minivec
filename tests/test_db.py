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


def test_ingest_skips_venv_git_and_hidden_dirs(tmp_path):
    (tmp_path / "notes.md").write_text("real content")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "more.txt").write_text("also real")
    for junk in (
        ".venv/lib/site-packages",
        ".git",
        "node_modules",
        "__pycache__",
        "minivec.egg-info",
    ):
        d = tmp_path / junk
        d.mkdir(parents=True)
        (d / "junk.txt").write_text("should be ignored")

    found = {p.name for p in MiniVec()._iter_text_files(tmp_path)}
    assert found == {"notes.md", "more.txt"}
