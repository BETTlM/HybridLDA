#!/usr/bin/env python3
"""CLI: python run_pipeline.py --n-docs 2000 --k 8,12,16 --corpora 20newsgroups,arxiv"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quiet import configure_warnings

configure_warnings()

from src.config import DEFAULT_K_GRID, DEFAULT_N_DOCS, EMBEDDING_MODEL, FULL_K_GRID, FULL_N_DOCS
from src.pipeline import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid LDA + semantic embeddings topic modeling")
    parser.add_argument("--n-docs", type=int, default=DEFAULT_N_DOCS, help="Documents per corpus")
    parser.add_argument(
        "--k",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_GRID),
        help="Comma-separated topic counts, e.g. 8,12,16 or 5,10,15,20,25,30",
    )
    parser.add_argument(
        "--corpora",
        type=str,
        default="20newsgroups,arxiv",
        help="Comma-separated: 20newsgroups,arxiv",
    )
    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--full", action="store_true", help="Use 8k docs and K=5..30")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    configure_warnings()
    n_docs = FULL_N_DOCS if args.full else args.n_docs
    k_grid = FULL_K_GRID if args.full else tuple(int(x) for x in args.k.split(",") if x.strip())
    corpora = tuple(c.strip() for c in args.corpora.split(",") if c.strip())
    run_experiments(
        corpora=corpora,
        n_docs=n_docs,
        k_grid=k_grid,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
