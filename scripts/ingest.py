"""Ingest earnings documents into the RAG knowledge base.

Usage:
    python -m scripts.ingest path/to/file.pdf [more.pdf ...] --ticker AAPL
    python -m scripts.ingest ./earnings_docs --ticker MSFT --doc-type 10-Q
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core import get_logger
from rag_pipeline import RAGPipeline
from rag_pipeline.ingestion import SUPPORTED_SUFFIXES

log = get_logger("ingest")


def _expand(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        else:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest.")
    parser.add_argument("--ticker", help="Associate documents with a ticker.")
    parser.add_argument("--doc-type", default="earnings", help="Document type metadata tag.")
    args = parser.parse_args()

    files = _expand(args.paths)
    if not files:
        log.error("No ingestible files found in: {}", args.paths)
        raise SystemExit(1)

    metadata = {"doc_type": args.doc_type}
    if args.ticker:
        metadata["ticker"] = args.ticker.upper()

    pipeline = RAGPipeline()
    total = pipeline.ingest_paths(files, metadata=metadata)
    log.info("Done. Ingested {} chunk(s) from {} file(s).", total, len(files))
    print(f"Ingested {total} chunk(s). Collection now holds {pipeline.store.count()}.")


if __name__ == "__main__":
    main()
