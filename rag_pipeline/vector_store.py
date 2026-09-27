"""Persistent ChromaDB vector store wrapper.

Embeddings are computed externally (Google Gemini) and supplied directly, so
Chroma is used purely as a persistent similarity index. Metadata filtering
(e.g. by ``ticker``) is supported through Chroma's ``where`` clause.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from core import get_logger, get_settings

log = get_logger("vector_store")

# ChromaDB's native (Rust) core must have exactly one PersistentClient per path
# per process — constructing a second client for the same path (and, on Windows,
# even for different paths) can crash the interpreter. Cache clients by resolved
# path so every consumer in the process shares one client instance.
_CLIENT_CACHE: dict[str, chromadb.api.ClientAPI] = {}


def _get_client(persist_dir: Path) -> chromadb.api.ClientAPI:
    key = str(persist_dir.resolve())
    client = _CLIENT_CACHE.get(key)
    if client is None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=key,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        _CLIENT_CACHE[key] = client
    return client


def _stable_id(text: str, metadata: dict[str, Any]) -> str:
    """Deterministic id so re-ingesting identical content upserts, not duplicates."""
    source = str(metadata.get("source", ""))
    chunk = str(metadata.get("chunk", ""))
    page = str(metadata.get("page", ""))
    digest = hashlib.sha256(f"{source}|{page}|{chunk}|{text}".encode()).hexdigest()
    return digest[:32]


class ChromaVectorStore:
    """A persistent collection of embedded document chunks.

    Args:
        persist_dir: Directory for the on-disk database. Defaults to config.
        collection_name: Collection to use. Defaults to config.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self._collection_name = collection_name or settings.chroma_collection

        self._client = _get_client(self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.debug(
            "Opened Chroma collection '{}' at {}", self._collection_name, self._persist_dir
        )

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

    def add(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Upsert chunks with precomputed embeddings.

        Args:
            texts: Chunk texts.
            embeddings: One vector per text.
            metadatas: One metadata dict per text.

        Returns:
            The list of stable chunk ids that were written.
        """
        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("texts, embeddings and metadatas must be the same length")
        if not texts:
            return []

        ids = [_stable_id(t, m) for t, m in zip(texts, metadatas, strict=True)]
        self._collection.upsert(
            ids=ids,
            documents=list(texts),
            embeddings=[list(e) for e in embeddings],
            metadatas=list(metadatas),
        )
        log.info("Upserted {} chunk(s) into '{}'", len(ids), self._collection_name)
        return ids

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` most similar chunks.

        Args:
            query_embedding: Embedding of the search query.
            top_k: Number of results. Defaults to configured ``retrieval_top_k``.
            where: Optional Chroma metadata filter, e.g. ``{"ticker": "AAPL"}``.

        Returns:
            A ranked list of ``{"text", "metadata", "score"}`` dicts where
            ``score`` is cosine similarity in ``[0, 1]`` (higher is better).
        """
        settings = get_settings()
        k = top_k or settings.retrieval_top_k
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(k, self.count()),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            hits.append(
                {
                    "text": text,
                    "metadata": metadata or {},
                    "score": round(1.0 - float(distance), 4),  # cosine distance -> similarity
                }
            )
        return hits

    def reset(self) -> None:
        """Delete and recreate the collection (irreversible)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.warning("Reset collection '{}'", self._collection_name)
