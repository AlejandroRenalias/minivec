"""Turn retrieved chunks into a written answer with an LLM (the "generation" in RAG).

Stage 4. This is the only part of minivec that talks to a paid API and needs
network access — the retrieval core (chunk/embed/store/index/db) stays offline
and free. ``answer()`` takes the results from ``MiniVec.query`` and asks Claude
to write a response grounded *only* in those passages.

Needs ``ANTHROPIC_API_KEY`` in the environment (or a .env file, see .env.example).
"""

from __future__ import annotations

from dataclasses import dataclass

from .db import SearchResult

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024

# Retrieved chunks scoring below this cosine similarity are treated as "nothing
# relevant found" — better to say "I don't know" than to answer from noise.
MIN_SCORE = 0.15

SYSTEM_PROMPT = (
    "You answer questions using only the context passages provided by the user. "
    "If the passages do not contain the answer, say you don't know — never fall "
    "back on outside knowledge. Name the source file(s) you used, in parentheses. "
    "Answer directly and concisely; do not narrate your reasoning."
)


@dataclass
class Answer:
    text: str
    sources: list[str]
    used_context: bool  # False => retrieval found nothing above the threshold


def _format_context(results: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] source: {r.chunk.source}\n{r.chunk.text}")
    return "\n\n".join(blocks)


def answer(
    question: str,
    results: list[SearchResult],
    *,
    client=None,
    model: str = MODEL,
    min_score: float = MIN_SCORE,
) -> Answer:
    """Write a grounded answer to ``question`` from retrieved ``results``.

    Args:
        question:  the user's question
        results:   output of ``MiniVec.query`` (already ranked, best first)
        client:    an ``anthropic.Anthropic`` instance; created from the
                   environment if not given (injected in tests)
        model:     Claude model id
        min_score: similarity floor; if no result clears it, returns an
                   "I don't know" answer without calling the API

    Returns:
        an :class:`Answer`
    """
    relevant = [r for r in results if r.score >= min_score]
    if not relevant:
        return Answer(
            text="I couldn't find anything relevant to that in the indexed documents.",
            sources=[],
            used_context=False,
        )

    if client is None:
        import anthropic  # lazy: keeps `import minivec` offline and dependency-free

        client = anthropic.Anthropic()

    user_content = (
        f"Context passages:\n\n{_format_context(relevant)}\n\n"
        f"---\nQuestion: {question}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # unique sources, preserving rank order
    sources = list(dict.fromkeys(r.chunk.source for r in relevant))
    return Answer(text=text, sources=sources, used_context=True)
