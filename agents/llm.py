"""Groq chat-model factory.

Centralises construction of the :class:`ChatGroq` client so model, temperature,
token limits and timeout come from a single validated configuration source.
"""

from __future__ import annotations

from langchain_groq import ChatGroq

from core import get_logger, get_settings

log = get_logger("llm")


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatGroq:
    """Construct a configured Groq chat model.

    Args:
        model: Override the configured Groq model. Must support tool calling.
        temperature: Override sampling temperature.
        max_tokens: Override maximum completion tokens.

    Returns:
        A ready-to-use :class:`ChatGroq` instance.
    """
    settings = get_settings()
    resolved_model = model or settings.groq_model
    log.debug("Creating ChatGroq client for model {}", resolved_model)
    return ChatGroq(
        model=resolved_model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        timeout=settings.llm_request_timeout,
        max_retries=2,
    )
