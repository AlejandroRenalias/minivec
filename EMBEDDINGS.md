# Embeddings — what I learned

The one idea minivec is built on, in plain terms. Written after working through
`explore.py` by predicting scores and checking them.

## The five things I can now explain

1. **Text gets turned into a list of numbers.** In minivec, 384 of them per
   passage. Think of that list as the *coordinates of a point* — like `(3, 5)`
   on a map, but in a space with 384 directions instead of 2.

2. **Similar meaning → points close together.** "a dog barked loudly" and "the
   puppy made a lot of noise" share no words, but their points are near each
   other because they *mean* similar things.

3. **Searching is just geometry.** minivec turns the question into a point and
   finds the stored points nearest to it. That's the entire mechanical job —
   no language understanding happens inside minivec itself.

4. **A model makes the numbers, and it's a black box.** `embed.py` runs a small
   neural network trained on a huge amount of text. It learned that words used
   in similar contexts ("walking the ___", "the ___ barked") belong at similar
   coordinates. I don't need to know how the training works to use it.

5. **It captures topic well, opposites badly.** "I love this movie" and "I hate
   this movie" scored **0.71** — high — even though the meaning is reversed,
   because both are *about having a strong opinion on a movie*. Negation and
   sentiment barely move the point.

## Why point 5 matters

Retrieval gets you to the right *neighborhood*, not the right *house*. Search
for "medicines safe in pregnancy" and the retriever might hand back a passage
about medicines that are **dangerous** in pregnancy — same topic, nearby point.

That's the reason the rest of the machinery exists:

- you still have to read the retrieved passages (or have the LLM read them carefully)
- chunking, `k`, re-ranking, and evaluation are all ways of coping with
  "close in topic ≠ correct answer"

## The scores I saw

| Pair | Score |
|---|---|
| "a dog barked loudly" vs "the puppy made a lot of noise" | 0.56 |
| "a dog barked loudly" vs "a cat sat quietly on the mat" | 0.36 |
| "a dog barked loudly" vs "I paid my electricity bill" | 0.09 |
| "how do I reset my password?" vs "I forgot my login credentials" | 0.68 |
| "I love this movie" vs "I hate this movie" | 0.71 |

Run `python explore.py` and edit the `PAIRS` list to try more.
