"""High-level RAG orchestration.

Ties ingestion, chunking, embedding and the vector store together behind two
verbs: :meth:`RAGPipeline.ingest` (write path) and
:meth:`RAGPipeline.retrieve` (read path).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import get_logger, get_settings
from rag_pipeline.chunking import chunk_documents
from rag_pipeline.embeddings import GoogleEmbeddingClient
from rag_pipeline.ingestion import Document, load_documents
from rag_pipeline.vector_store import ChromaVectorStore

log = get_logger("rag_pipeline")


@dataclass(slots=True)
class RetrievedChunk:
    """A single retrieved passage with its relevance score."""

    text: str
    metadata: dict[str, Any]
    score: float

    @property
    def citation(self) -> str:
        """Human-readable provenance label for the passage."""
        source = self.metadata.get("source", "unknown")
        page = self.metadata.get("page")
        return f"{source} (p.{page})" if page else str(source)


class RAGPipeline:
    """Orchestrates the earnings-document knowledge base.

    Args:
        vector_store: Injected store (defaults to the configured Chroma store).
        embedding_client: Injected embedder (defaults to Google Gemini).
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedding_client: GoogleEmbeddingClient | None = None,
    ) -> None:
        self._store = vector_store or ChromaVectorStore()
        self._embedder = embedding_client or GoogleEmbeddingClient()

    @property
    def store(self) -> ChromaVectorStore:
        return self._store

    def ingest_documents(self, documents: list[Document]) -> int:
        """Chunk, embed and store already-loaded documents.

        Returns:
            The number of chunks written.
        """
        if not documents:
            return 0
        chunks = chunk_documents(documents)
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = self._embedder.embed_documents(texts)
        self._store.add(texts=texts, embeddings=embeddings, metadatas=metadatas)
        log.info("Ingested {} chunk(s); collection now holds {}", len(chunks), self._store.count())
        return len(chunks)

    def ingest_paths(
        self, paths: Iterable[str | Path], metadata: dict[str, Any] | None = None
    ) -> int:
        """Load files from disk then ingest them.

        Args:
            paths: Files to ingest (PDF / text / markdown).
            metadata: Shared metadata (e.g. ``{"ticker": "AAPL"}``).
        """
        documents = load_documents(paths, metadata)
        return self.ingest_documents(documents)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        ticker: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant passages for a query.

        Args:
            query: Natural-language question.
            top_k: Number of passages (defaults to configured value).
            ticker: Optional ticker filter (matches ingest-time metadata).
        """
        settings = get_settings()
        k = top_k or settings.retrieval_top_k
        where = {"ticker": ticker.upper()} if ticker else None

        query_embedding = self._embedder.embed_query(query)
        hits = self._store.query(query_embedding, top_k=k, where=where)
        return [
            RetrievedChunk(text=h["text"], metadata=h["metadata"], score=h["score"])
            for h in hits
        ]

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Render retrieved chunks into a citation-annotated context block."""
        if not chunks:
            return "No relevant earnings-document passages were found."
        return "\n\n".join(
            f"[{i}] {chunk.citation}\n{chunk.text}" for i, chunk in enumerate(chunks, start=1)
        )
