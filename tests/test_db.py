"""End-to-end tests for minivec.db — filled in during stage 2."""

import pytest

from minivec.db import MiniVec


@pytest.mark.skip(reason="stage 2")
def test_ingest_then_query_roundtrip(tmp_path):
    (tmp_path / "a.md").write_text("The auth flow uses short-lived OAuth tokens.")
    (tmp_path / "b.md").write_text("The kitchen was painted a pale shade of green.")

    db = MiniVec()
    n = db.ingest(tmp_path)
    assert n >= 2

    results = db.query("how does login work?", k=1)
    assert results[0].chunk.source.endswith("a.md")
