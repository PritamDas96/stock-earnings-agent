"""End-to-end connectivity and configuration health check.

Verifies configuration, the Groq LLM, Google embeddings and the vector store are
all reachable and correctly configured. Exits non-zero if any check fails, so it
is suitable as a container/CI readiness probe.

Usage:
    python -m scripts.healthcheck
"""

from __future__ import annotations

import sys

from core import get_logger, get_settings

log = get_logger("healthcheck")


def _check(name: str, fn) -> bool:
    try:
        detail = fn()
        print(f"  [OK]   {name}: {detail}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {name}: {exc}")
        return False


def _check_config() -> str:
    settings = get_settings()
    return f"model={settings.groq_model}, embedding={settings.embedding_model}"


def _check_embeddings() -> str:
    from rag_pipeline import GoogleEmbeddingClient

    vector = GoogleEmbeddingClient().embed_query("health check")
    return f"{len(vector)}-dimensional vector"


def _check_llm() -> str:
    from agents.llm import get_llm

    response = get_llm(max_tokens=16).invoke("Reply with the single word: ok")
    return f"responded ({response.content.strip()[:20]!r})"


def _check_vector_store() -> str:
    from rag_pipeline import ChromaVectorStore

    store = ChromaVectorStore()
    return f"collection holds {store.count()} chunk(s)"


def main() -> None:
    print("stock-earnings-agent health check")
    print("-" * 40)
    results = [
        _check("configuration", _check_config),
        _check("google embeddings", _check_embeddings),
        _check("vector store", _check_vector_store),
        _check("groq llm", _check_llm),
    ]
    print("-" * 40)
    if all(results):
        print("All systems operational.")
        sys.exit(0)
    print("One or more checks failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
