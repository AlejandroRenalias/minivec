"""Command-line interface: ``minivec ingest`` / ``query`` / ``ask``.

    minivec ingest <path> [--db ./store] [--chunk-size 500] [--overlap 50]
    minivec query  "<text>" [--db ./store] [-k 5] [--index brute|ivf]
    minivec ask    "<question>" [--db ./store] [-k 5] [--index brute|ivf]
                   [--min-score 0.15] [--model claude-sonnet-5] [--show-context]

`ingest` and `query` are offline and free. `ask` sends the retrieved chunks to
Claude for a grounded answer and needs ANTHROPIC_API_KEY (see .env.example).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .db import MiniVec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minivec", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a folder of text files")
    p_ingest.add_argument("path", help="file or directory to index")
    p_ingest.add_argument("--db", default="./store", help="store directory to create")
    p_ingest.add_argument("--chunk-size", type=int, default=500)
    p_ingest.add_argument("--overlap", type=int, default=50)

    p_query = sub.add_parser("query", help="search an existing store")
    p_query.add_argument("text", help="the question / search text")
    p_query.add_argument("--db", default="./store", help="store directory to read")
    p_query.add_argument("-k", type=int, default=5, help="number of results")
    p_query.add_argument("--index", choices=["brute", "ivf"], default="brute")

    p_ask = sub.add_parser("ask", help="retrieve, then get a grounded LLM answer")
    p_ask.add_argument("text", help="the question")
    p_ask.add_argument("--db", default="./store", help="store directory to read")
    p_ask.add_argument("-k", type=int, default=5, help="chunks to retrieve")
    p_ask.add_argument("--index", choices=["brute", "ivf"], default="brute")
    p_ask.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="similarity floor; below it, answer 'I don't know' without calling the API",
    )
    p_ask.add_argument("--model", default=None, help="Claude model id")
    p_ask.add_argument(
        "--show-context",
        action="store_true",
        help="also print the retrieved chunks that were sent to the model",
    )

    return parser


def _cmd_ingest(args: argparse.Namespace) -> int:
    db = MiniVec.load(args.db) if Path(args.db).exists() else MiniVec()
    added = db.ingest(args.path, chunk_size=args.chunk_size, overlap=args.overlap)
    db.save(args.db)
    total = len(db.store) if db.store is not None else 0
    if added == 0:
        print(f"No text files found under {args.path!r}.")
    else:
        print(f"Indexed {added} chunks from {args.path!r} -> {args.db} ({total} total).")
    return 0


def _load_db(args: argparse.Namespace) -> MiniVec | None:
    if not Path(args.db).exists():
        print(f"No store at {args.db!r}. Run `minivec ingest` first.")
        return None
    return MiniVec.load(args.db, index=args.index)


def _cmd_query(args: argparse.Namespace) -> int:
    db = _load_db(args)
    if db is None:
        return 1
    results = db.query(args.text, k=args.k)
    if not results:
        print("No results.")
        return 0
    for rank, result in enumerate(results, 1):
        snippet = " ".join(result.chunk.text.split())[:200]
        print(f"{rank}. ({result.score:.2f}) {result.chunk.source}")
        print(f"   {snippet}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    db = _load_db(args)
    if db is None:
        return 1

    from .generate import MIN_SCORE, MODEL, answer

    results = db.query(args.text, k=args.k)
    if args.show_context:
        for rank, result in enumerate(results, 1):
            snippet = " ".join(result.chunk.text.split())[:200]
            print(f"  [{rank}] ({result.score:.2f}) {result.chunk.source}")
            print(f"      {snippet}")
        print()

    try:
        result = answer(
            args.text,
            results,
            model=args.model or MODEL,
            min_score=args.min_score if args.min_score is not None else MIN_SCORE,
        )
    except Exception as exc:  # missing key, network, API error
        print(f"Generation failed: {exc}")
        print("Check ANTHROPIC_API_KEY (see .env.example).")
        return 1

    print(result.text)
    if result.sources:
        print("\nSources: " + ", ".join(result.sources))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "query":
        return _cmd_query(args)
    if args.command == "ask":
        return _cmd_ask(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
