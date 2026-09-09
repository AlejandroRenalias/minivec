# minivec — the plain-English map

Read this once. It's the "shape of how it works," no jargon. Details live in the
code and the [README](README.md); this is just the mental model.

## The whole thing in two sentences

minivec takes your text files, cuts them into small passages, and turns each
passage into a list of numbers that captures its **meaning**. When you ask a
question, it turns the question into numbers too, finds the stored passages whose
numbers are most similar, and — if you use `ask` — hands those passages to Claude
to write an answer grounded in them.

## The one idea everything rests on

**A piece of text can be turned into a list of numbers, and text that means
similar things gets similar numbers.**

That list of numbers is called an *embedding* or a *vector*. In minivec each one
is 384 numbers long. Think of it as a "meaning fingerprint." Two passages about
the same topic have close fingerprints even if they share no actual words.

Once text is fingerprints, "find related passages" becomes "find the closest
fingerprints" — a math problem instead of a language problem.

## The six pieces

| File | Its one job | When it runs |
|---|---|---|
| `chunk.py` | Cut big documents into small overlapping passages. Smaller passages = more precise search. | during `ingest` |
| `embed.py` | Turn a passage into its 384-number fingerprint, using a small AI model on your laptop. **Everything depends on this.** | `ingest` and `query` |
| `store.py` | Keep all the fingerprints in one big table, remember which file each came from, save it all to disk. | `ingest` (write), `query` (read) |
| `index.py` | Given the question's fingerprint, return the handful of stored fingerprints closest to it. Two versions: `BruteForceIndex` (check every one, exact) and `IVFIndex` (sort into buckets first, check only nearby buckets, faster, occasionally misses one). | `query` |
| `db.py` | The conductor. On `ingest` it calls chunk → embed → store. On `query` it calls embed → index and returns the matching passages. | always |
| `generate.py` | Send the matched passages plus the question to Claude, get back a written answer that sticks to those passages. The only part that goes online / costs money. | `ask` only |

## The two flows

```
minivec ingest <folder>
    folder of .txt/.md files
      → chunk.py    cut into passages
      → embed.py    each passage → fingerprint
      → store.py    save fingerprints + which file each came from

minivec query "question"          (or `ask`, which adds one step)
    "question"
      → embed.py    question → fingerprint
      → index.py    find the closest stored fingerprints
      → db.py       return those passages, best first
      → generate.py (ask only) Claude writes an answer from them
```

## What you do NOT need to hold in your head

- how the embedding model was trained (it just works, treat it as a black box)
- the k-means math inside `IVFIndex`
- the disk file formats in `store.py`
- cosine similarity vs dot product (for minivec they're the same thing, because
  every fingerprint is scaled to length 1)

If someone asks "how does it work," the two-sentence summary plus "similar
meaning → similar numbers → find the closest numbers" is a complete answer.
