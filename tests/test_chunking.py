"""Tests for text chunking (offline, no network)."""

from __future__ import annotations

from rag_pipeline.chunking import chunk_documents, chunk_text
from rag_pipeline.ingestion import Document


def test_chunk_text_respects_size():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_short_input_single_chunk():
    chunks = chunk_text("short text", chunk_size=100, chunk_overlap=0)
    assert chunks == ["short text"]


def test_chunk_documents_preserves_and_indexes_metadata():
    doc = Document(content="sentence one. " * 100, metadata={"ticker": "AAPL", "source": "x.txt"})
    chunks = chunk_documents([doc], chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["ticker"] == "AAPL"
        assert chunk.metadata["source"] == "x.txt"
        assert chunk.metadata["chunk"] == index


def test_chunk_documents_empty_input():
    assert chunk_documents([]) == []
