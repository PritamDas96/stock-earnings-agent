"""Shared pytest fixtures.

Ensures tests run offline and deterministically: dummy credentials are injected
so :func:`core.get_settings` validates without a real ``.env``, and the settings
cache is cleared between tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _dummy_env(monkeypatch):
    """Provide dummy credentials and clear the settings cache for each test."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    from core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeEmbeddingClient:
    """Deterministic, offline stand-in for :class:`GoogleEmbeddingClient`.

    Produces small fixed-dimension vectors derived from character codes so
    similar text yields similar vectors — enough for store/retrieve tests.
    """

    dimension = 8

    def _vec(self, text: str) -> list[float]:
        buckets = [0.0] * self.dimension
        for i, char in enumerate(text.lower()):
            buckets[i % self.dimension] += (ord(char) % 13) / 13.0
        norm = sum(v * v for v in buckets) ** 0.5 or 1.0
        return [v / norm for v in buckets]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts) -> list[list[float]]:
        return [self._vec(t) for t in texts]


@pytest.fixture
def fake_embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture(scope="session")
def chroma_dir(tmp_path_factory):
    """One shared on-disk Chroma path for the whole test session.

    ChromaDB's native core must have a single client per process, so all
    vector-store tests share this directory and isolate via unique collection
    names instead of separate paths.
    """
    return tmp_path_factory.mktemp("chroma_store")


@pytest.fixture
def collection_name(request) -> str:
    """A unique collection name derived from the running test."""
    return f"test_{abs(hash(request.node.nodeid)) % 10_000_000}"
