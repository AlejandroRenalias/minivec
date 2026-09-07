# minivec

A from-scratch miniature **vector database** / semantic search engine, built to learn
how the retrieval layer behind RAG actually works.

## The idea

Point it at a folder of text files. It splits them into chunks, turns each chunk into a
vector with an embedding model, and stores those vectors. Ask a question in plain
English and it returns the chunks whose meaning is closest to the question.

```
$ minivec ingest ./my-notes
Indexed 412 chunks from 37 files.

$ minivec query "what did I decide about the auth flow?"
1. (0.83) notes/2026-03-meeting.md
   "We agreed to drop password login and go OAuth-only..."
2. (0.79) notes/auth-design.md
   "The auth flow uses short-lived tokens refreshed every..."
```

Real vector databases (Pinecone, Chroma, pgvector) already exist. The point of building
one is to understand, concretely:

- what an embedding is, and why similar meanings end up as nearby vectors
- how "closeness" is computed (cosine similarity, dot product, why normalization matters)
- why brute-force search doesn't scale, and what an **index** does about it
- how to *measure* whether retrieval is any good (**recall** vs. **speed**)

## How it works

```
text file ──chunk──▶ "...paragraph..." ──embed──▶ [0.03, -0.81, ...]  (a 384-dim vector)
                                                          │
                                                          ▼
                                        stored in a Store (vectors + metadata)
                                                          │
query "..." ──embed──▶ [ ... ] ──────▶ Index.search() ────┘──▶ top-k closest chunks
```

## Architecture

| Module | Responsibility |
|---|---|
| `minivec/chunk.py` | Split raw text into overlapping `Chunk`s (text + source + offset) |
| `minivec/embed.py` | Wrap a local `sentence-transformers` model: `list[str] -> np.ndarray` |
| `minivec/store.py` | Hold the vector matrix + parallel metadata; save/load a Store directory |
| `minivec/index.py` | Search strategies: `BruteForceIndex`, `IVFIndex` (same interface) |
| `minivec/db.py` | `MiniVec` — ties it together: `ingest(path)`, `query(text, k)` |
| `minivec/cli.py` | `minivec ingest` / `minivec query` |
| `benchmark.py` | IVF vs. brute force: recall@k and query latency |

### Key design choices

- Vectors are stored as one `float32` `np.ndarray` of shape `(n, dim)`, **L2-normalized**
  so a plain dot product equals cosine similarity.
- Metadata is a parallel `list[Chunk]` — row `i` of the matrix describes chunk `i`.
- A Store on disk is a directory: `vectors.npy`, `meta.jsonl`, `config.json`
  (model name, dimension, index type).
- Every index implements the same tiny interface:

  ```python
  class Index(Protocol):
      def build(self, vectors: np.ndarray) -> None: ...
      def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
          """Return (scores, ids) of the k best matches, best first."""
  ```

- **IVF** (inverted file index): k-means the vectors into `nlist` clusters; each query
  only compares against vectors in its `nprobe` nearest clusters. Fewer comparisons,
  slightly lower recall — and the benchmark quantifies exactly that.

## Build stages

- [x] **0** — scaffold + design
- [x] **1** — `chunk` + `embed` (+ tests)
- [x] **2** — `store` + `BruteForceIndex` + `db` + CLI (end-to-end `ingest`/`query`)
- [x] **3** — `IVFIndex` + `benchmark.py`
- [ ] **4** — polish: more tests, usage docs (optional)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

The first `ingest` downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB) once,
then runs fully offline.

## Usage

```bash
# index a folder of .txt / .md / .rst files into a store directory
minivec ingest ./my-notes --db ./notes-store

# ask it questions
minivec query "what did I decide about the auth flow?" --db ./notes-store -k 5
```

`ingest` is incremental — run it again with more files and the same `--db` to add
to the existing store. Each result line is `rank. (score) source` followed by a
snippet, where `score` is cosine similarity in `[-1, 1]` (higher = closer).

## Benchmark: why an index

`python benchmark.py` builds a 200k-vector synthetic set, treats brute force as
ground truth, and sweeps IVF's `nprobe`:

```
 nprobe   recall@k   ms/query   speedup
      1      0.743      0.050     57.8x
      2      0.938      0.065     44.1x
      4      0.997      0.095     30.3x
      8      1.000      0.227     12.7x
     64      1.000      1.917      1.5x
    128      1.000      4.555      0.6x
```

The lesson in one table: probing 4 of 1024 clusters returns 99.7% of the true
top-10 for ~30x less work. Beyond a point, more probing buys no recall and just
costs time — at `nprobe=128` IVF is *slower* than brute force, because gathering
candidate rows in Python outweighs the matmul it saved. Real vector databases
live at that recall/latency knee; this benchmark lets you see it move.

## Glossary

- **Embedding** — a fixed-length list of numbers representing a piece of text's meaning.
- **Dimension** — how many numbers in each vector (384 for the default model).
- **Cosine similarity** — the angle between two vectors; 1.0 = identical direction.
- **Recall@k** — of the *true* k nearest chunks, how many the index actually returned.
- **IVF / nprobe** — cluster the vectors; only search the `nprobe` closest clusters.
