"""Command-line interface: ``minivec ingest`` and ``minivec query``.

Stage 2.

    minivec ingest <path> [--db ./store] [--chunk-size 500] [--overlap 50]
    minivec query "<text>" [--db ./store] [-k 5] [--index brute|ivf]
"""

from __future__ import annotations

import argparse


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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError("stage 2")


if __name__ == "__main__":
    raise SystemExit(main())
