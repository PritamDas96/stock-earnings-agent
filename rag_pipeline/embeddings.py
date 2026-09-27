"""Google Gemini embeddings via the Generative Language REST API.

Uses direct ``requests`` calls (per project decision) rather than
``sentence-transformers``/PyTorch, which are DLL-incompatible with Python 3.13
on Windows. Requests are retried with exponential backoff on transient network
and 5xx errors.
"""

from __future__ import annotations

from collections.abc import Sequence

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core import get_logger, get_settings

log = get_logger("embeddings")

# Task type hints let Gemini optimise the vector for its intended use.
TASK_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


class EmbeddingError(RuntimeError):
    """Raised when the embedding API returns an unrecoverable error."""


class GoogleEmbeddingClient:
    """Thin, resilient client for ``gemini-embedding-001``.

    Args:
        api_key: Google API key. Defaults to the configured ``GOOGLE_API_KEY``.
        model: Embedding model name. Defaults to the configured model.
        timeout: Per-request timeout in seconds.
        batch_size: Maximum number of texts per batch request.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        batch_size: int | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.google_api_key
        self._model = model or settings.embedding_model
        self._timeout = timeout or settings.http_timeout
        self._batch_size = batch_size or settings.embedding_batch_size
        self._dimension = settings.embedding_dimension
        self._single_url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{self._model}:embedContent"
        )
        self._batch_url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{self._model}:batchEmbedContents"
        )
        self._session = requests.Session()

    @property
    def dimension(self) -> int:
        """Dimensionality of vectors produced by this client."""
        return self._dimension

    # -- Public API -------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self._embed_one(text, TASK_RETRIEVAL_QUERY)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents, preserving input order.

        Empty input returns an empty list. Batches larger than ``batch_size``
        are split transparently.
        """
        cleaned = [t for t in texts]
        if not cleaned:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), self._batch_size):
            batch = cleaned[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch, TASK_RETRIEVAL_DOCUMENT))
        log.debug("Embedded {} documents into {}-d vectors", len(vectors), self._dimension)
        return vectors

    # -- Internal ---------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((requests.RequestException, EmbeddingError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _embed_one(self, text: str, task_type: str) -> list[float]:
        payload = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        response = self._session.post(
            self._single_url,
            params={"key": self._api_key},
            json=payload,
            timeout=self._timeout,
        )
        self._raise_for_status(response)
        data = response.json()
        try:
            return data["embedding"]["values"]
        except (KeyError, TypeError) as exc:  # pragma: no cover - defensive
            raise EmbeddingError(f"Unexpected embedding response: {data}") from exc

    @retry(
        retry=retry_if_exception_type((requests.RequestException, EmbeddingError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _embed_batch(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        payload = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                for text in texts
            ]
        }
        response = self._session.post(
            self._batch_url,
            params={"key": self._api_key},
            json=payload,
            timeout=self._timeout,
        )
        self._raise_for_status(response)
        data = response.json()
        try:
            return [item["values"] for item in data["embeddings"]]
        except (KeyError, TypeError) as exc:  # pragma: no cover - defensive
            raise EmbeddingError(f"Unexpected batch embedding response: {data}") from exc

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.status_code == 200:
            return
        # Retry on server errors and rate limits; fail fast on client errors.
        if response.status_code >= 500 or response.status_code == 429:
            raise EmbeddingError(
                f"Transient embedding API error {response.status_code}: {response.text[:300]}"
            )
        raise EmbeddingError(
            f"Embedding API error {response.status_code}: {response.text[:300]}"
        )
