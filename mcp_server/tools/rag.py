"""RAG retrieval tool over the ingested earnings-document knowledge base.

Wraps :class:`RAGPipeline` for use as an MCP tool and a LangChain agent tool.
The pipeline is lazily instantiated so importing this module never triggers
network calls or database initialisation.
"""

from __future__ import annotations

from typing import Any

from core import get_logger
from rag_pipeline import RAGPipeline

log = get_logger("rag_tool")

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def search_earnings_documents(
    query: str, ticker: str | None = None, top_k: int = 5
) -> dict[str, Any]:
    """Semantic search over ingested earnings documents / filings.

    Args:
        query: Natural-language question about a company's earnings.
        ticker: Optional ticker to restrict results to one company.
        top_k: Number of passages to return.

    Returns:
        A dict with the formatted ``context``, structured ``passages`` and the
        ``count`` of matches (or ``{"error": ...}`` on failure).
    """
    if not query or not query.strip():
        return {"error": "query is required"}

    try:
        pipeline = _get_pipeline()
        chunks = pipeline.retrieve(query, top_k=top_k, ticker=ticker)
    except Exception as exc:  # noqa: BLE001 - surface as data to the caller
        log.warning("search_earnings_documents failed: {}", exc)
        return {"error": f"Retrieval failed: {exc}"}

    return {
        "query": query,
        "ticker": ticker.upper() if ticker else None,
        "count": len(chunks),
        "context": pipeline.format_context(chunks),
        "passages": [
            {"text": c.text, "citation": c.citation, "score": c.score} for c in chunks
        ],
    }
