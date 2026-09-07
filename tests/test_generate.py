"""Tests for minivec.generate — the LLM call is faked, so these are offline."""

from types import SimpleNamespace

from minivec.chunk import Chunk
from minivec.db import SearchResult
from minivec.generate import answer


class FakeMessages:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        self._outer.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._outer.reply)]
        )


class FakeClient:
    """Stands in for anthropic.Anthropic() — records calls, returns a fixed reply."""

    def __init__(self, reply="Grounded answer. (auth.md)"):
        self.reply = reply
        self.calls: list[dict] = []
        self.messages = FakeMessages(self)


def _result(text, source, score):
    return SearchResult(chunk=Chunk(text=text, source=source, start=0), score=score)


def test_answer_calls_model_with_context_and_question():
    client = FakeClient()
    results = [
        _result("OAuth tokens, refreshed hourly.", "auth.md", 0.71),
        _result("The kitchen is green.", "kitchen.md", 0.22),
    ]

    out = answer("how does login work?", results, client=client, min_score=0.15)

    assert out.used_context is True
    assert out.text == "Grounded answer. (auth.md)"
    assert out.sources == ["auth.md", "kitchen.md"]

    assert len(client.calls) == 1
    sent = client.calls[0]["messages"][0]["content"]
    assert "OAuth tokens" in sent
    assert "how does login work?" in sent
    assert "only" in client.calls[0]["system"]


def test_low_scores_skip_the_api_entirely():
    client = FakeClient()
    results = [_result("unrelated text", "misc.md", 0.04)]

    out = answer("something off-topic", results, client=client, min_score=0.15)

    assert out.used_context is False
    assert out.sources == []
    assert "couldn't find anything relevant" in out.text
    assert client.calls == []  # no paid call made


def test_sources_are_deduped_in_rank_order():
    client = FakeClient()
    results = [
        _result("chunk one", "doc.md", 0.6),
        _result("chunk two", "doc.md", 0.5),
        _result("chunk three", "other.md", 0.4),
    ]

    out = answer("q", results, client=client, min_score=0.15)
    assert out.sources == ["doc.md", "other.md"]


def test_empty_results_means_no_context():
    client = FakeClient()
    out = answer("q", [], client=client)
    assert out.used_context is False
    assert client.calls == []
