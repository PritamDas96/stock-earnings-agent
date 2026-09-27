"""Retrieval-augmented generation pipeline.

Turns raw earnings documents (PDF / text) into a queryable knowledge base:

    ingest  →  chunk  →  embed (Google Gemini)  →  store (ChromaDB)
    query   →  embed  →  similarity search  →  ranked context

The public surface is the :class:`RAGPipeline` orchestrator plus the individual
building blocks for advanced use and testing.
"""

from rag_pipeline.chunking import chunk_documents, chunk_text
from rag_pipeline.embeddings import GoogleEmbeddingClient
from rag_pipeline.ingestion import Document, load_documents, load_path
from rag_pipeline.pipeline import RAGPipeline, RetrievedChunk
from rag_pipeline.vector_store import ChromaVectorStore

__all__ = [
    "GoogleEmbeddingClient",
    "Document",
    "load_documents",
    "load_path",
    "chunk_documents",
    "chunk_text",
    "ChromaVectorStore",
    "RAGPipeline",
    "RetrievedChunk",
]
