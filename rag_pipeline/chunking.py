"""Text chunking using LangChain's recursive character splitter.

Documents are split into overlapping windows sized for the embedding model's
context, preserving per-chunk metadata (and adding a ``chunk`` index) so that
retrieved passages remain traceable to their source.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import get_logger, get_settings
from rag_pipeline.ingestion import Document

log = get_logger("chunking")


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split raw text into overlapping chunks."""
    settings = get_settings()
    splitter = _build_splitter(
        chunk_size or settings.chunk_size,
        chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )
    return splitter.split_text(text)


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into chunk-level documents.

    Each output document inherits its parent's metadata and gains a ``chunk``
    index (0-based within the parent).
    """
    settings = get_settings()
    splitter = _build_splitter(
        chunk_size or settings.chunk_size,
        chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )

    chunks: list[Document] = []
    for document in documents:
        pieces = splitter.split_text(document.content)
        for index, piece in enumerate(pieces):
            chunks.append(
                Document(content=piece, metadata={**document.metadata, "chunk": index})
            )

    log.info("Split {} document(s) into {} chunk(s)", len(documents), len(chunks))
    return chunks
