"""Centralised, validated application configuration.

All runtime configuration is read once from environment variables (and the
project-level ``.env`` file) into a strongly-typed :class:`Settings` object.
Every other module obtains configuration through :func:`get_settings`, which
is cached so the ``.env`` file is parsed a single time per process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root — resolved relative to this file so paths work regardless of
# the current working directory (important for the MCP server and Streamlit).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings sourced from the environment / ``.env``.

    Field names map case-insensitively to environment variables, e.g. the
    field ``groq_api_key`` is populated from ``GROQ_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Credentials (required) ------------------------------------------
    groq_api_key: str = Field(..., description="API key for Groq inference.")
    google_api_key: str = Field(
        ..., description="API key for Google Generative Language (embeddings)."
    )

    # --- Large language model --------------------------------------------
    groq_model: str = Field(
        "qwen/qwen3.8-27b",
        description="Groq chat model. Must support tool calling.",
    )
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    # Kept modest so a single request stays within Groq free-tier per-request
    # output-token limits. Raise via LLM_MAX_TOKENS on a paid (Dev) tier.
    llm_max_tokens: int = Field(800, gt=0)
    llm_request_timeout: float = Field(90.0, gt=0)

    # --- Embeddings ------------------------------------------------------
    embedding_model: str = Field("gemini-embedding-001")
    embedding_dimension: int = Field(3072, gt=0)
    embedding_batch_size: int = Field(16, gt=0)

    # --- Vector store ----------------------------------------------------
    chroma_persist_dir: Path = Field(PROJECT_ROOT / "chroma_db")
    chroma_collection: str = Field("earnings_documents")

    # --- Chunking --------------------------------------------------------
    chunk_size: int = Field(1000, gt=0)
    chunk_overlap: int = Field(150, ge=0)

    # --- Retrieval -------------------------------------------------------
    retrieval_top_k: int = Field(5, gt=0)

    # --- Outbound HTTP ---------------------------------------------------
    http_timeout: float = Field(30.0, gt=0)
    sec_user_agent: str = Field(
        "StockEarningsAgent/1.0 (contact@stockagent.example)",
        description="SEC EDGAR requires a descriptive User-Agent header.",
    )

    # --- Logging ---------------------------------------------------------
    log_level: str = Field("INFO")

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        valid = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}")
        return level

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_below_size(cls, value: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and value >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @property
    def embedding_endpoint(self) -> str:
        """Single-content embedding REST endpoint."""
        return (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"{self.embedding_model}:embedContent"
        )

    @property
    def embedding_batch_endpoint(self) -> str:
        """Batch embedding REST endpoint."""
        return (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"{self.embedding_model}:batchEmbedContents"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated application settings.

    Raises:
        pydantic.ValidationError: If required credentials are missing or a
            value fails validation. This surfaces misconfiguration loudly at
            startup rather than deep inside a request.
    """
    return Settings()  # type: ignore[call-arg]
