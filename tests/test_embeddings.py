"""Tests for the Google embeddings client (network mocked)."""

from __future__ import annotations

import pytest

from rag_pipeline.embeddings import EmbeddingError, GoogleEmbeddingClient


class _Response:
    def __init__(self, status_code: int, payload: dict, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self):
        return self._payload


def test_embed_query_returns_vector(monkeypatch):
    client = GoogleEmbeddingClient()

    def fake_post(url, params, json, timeout):
        return _Response(200, {"embedding": {"values": [0.1, 0.2, 0.3]}})

    monkeypatch.setattr(client._session, "post", fake_post)
    assert client.embed_query("hello") == [0.1, 0.2, 0.3]


def test_embed_documents_batches(monkeypatch):
    client = GoogleEmbeddingClient(batch_size=2)
    calls = {"n": 0}

    def fake_post(url, params, json, timeout):
        calls["n"] += 1
        count = len(json["requests"])
        return _Response(200, {"embeddings": [{"values": [1.0, 2.0]} for _ in range(count)]})

    monkeypatch.setattr(client._session, "post", fake_post)
    vectors = client.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert calls["n"] == 2  # 3 items / batch_size 2 -> two batches


def test_client_error_raises(monkeypatch):
    client = GoogleEmbeddingClient()

    def fake_post(url, params, json, timeout):
        return _Response(400, {"error": "bad request"}, text="bad request")

    monkeypatch.setattr(client._session, "post", fake_post)
    with pytest.raises(EmbeddingError):
        client.embed_query("hello")


def test_empty_documents_returns_empty():
    assert GoogleEmbeddingClient().embed_documents([]) == []
