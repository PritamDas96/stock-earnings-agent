"""Tests for the Chroma vector store and RAG pipeline (offline embeddings).

These are marked ``integration`` because they exercise ChromaDB's native (Rust)
core. On Windows that core conflicts with pandas/pyarrow when both run in one
process, so these tests are run in a dedicated process (see the Makefile
``test-integration`` target) and are excluded from the default unit-test run.
"""

from __future__ import annotations

import pytest

from rag_pipeline.ingestion import Document
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.vector_store import ChromaVectorStore

pytestmark = pytest.mark.integration


def _store(chroma_dir, collection_name) -> ChromaVectorStore:
    return ChromaVectorStore(persist_dir=chroma_dir, collection_name=collection_name)


def test_add_and_query_roundtrip(chroma_dir, collection_name, fake_embedder):
    store = _store(chroma_dir, collection_name)
    texts = ["alpha revenue growth", "beta margin decline"]
    embeddings = fake_embedder.embed_documents(texts)
    metadatas = [{"source": "a.txt", "chunk": 0}, {"source": "b.txt", "chunk": 0}]

    ids = store.add(texts, embeddings, metadatas)
    assert len(ids) == 2
    assert store.count() == 2

    hits = store.query(fake_embedder.embed_query("alpha revenue growth"), top_k=1)
    assert len(hits) == 1
    assert 0.0 <= hits[0]["score"] <= 1.0


def test_add_is_idempotent(chroma_dir, collection_name, fake_embedder):
    store = _store(chroma_dir, collection_name)
    texts = ["same text"]
    embeddings = fake_embedder.embed_documents(texts)
    metadatas = [{"source": "a.txt", "chunk": 0}]

    store.add(texts, embeddings, metadatas)
    store.add(texts, embeddings, metadatas)  # re-add identical content
    assert store.count() == 1  # stable id -> upsert, not duplicate


def test_length_mismatch_raises(chroma_dir, collection_name, fake_embedder):
    store = _store(chroma_dir, collection_name)
    with pytest.raises(ValueError):
        store.add(["a", "b"], fake_embedder.embed_documents(["a"]), [{"source": "x"}])


def test_query_empty_store_returns_empty(chroma_dir, collection_name, fake_embedder):
    store = _store(chroma_dir, collection_name)
    assert store.query(fake_embedder.embed_query("q")) == []


def test_metadata_filter(chroma_dir, collection_name, fake_embedder):
    store = _store(chroma_dir, collection_name)
    texts = ["apple revenue", "microsoft revenue"]
    store.add(texts, fake_embedder.embed_documents(texts),
              [{"source": "a", "chunk": 0, "ticker": "AAPL"},
               {"source": "b", "chunk": 0, "ticker": "MSFT"}])

    hits = store.query(fake_embedder.embed_query("revenue"), top_k=5, where={"ticker": "AAPL"})
    assert all(h["metadata"]["ticker"] == "AAPL" for h in hits)


def test_pipeline_ingest_and_retrieve(chroma_dir, collection_name, fake_embedder):
    store = _store(chroma_dir, collection_name)
    pipeline = RAGPipeline(vector_store=store, embedding_client=fake_embedder)

    doc = Document(content="Revenue grew strongly on Services. " * 5,
                   metadata={"source": "call.txt", "ticker": "AAPL"})
    n = pipeline.ingest_documents([doc])
    assert n >= 1

    chunks = pipeline.retrieve("How did revenue grow?", top_k=2, ticker="AAPL")
    assert chunks
    assert "call.txt" in pipeline.format_context(chunks)
