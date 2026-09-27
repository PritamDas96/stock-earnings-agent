"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings, get_settings


def test_settings_load_from_env():
    settings = get_settings()
    assert settings.groq_api_key == "test-groq-key"
    assert settings.google_api_key == "test-google-key"
    assert settings.embedding_dimension == 3072
    assert settings.groq_model  # non-empty default


def test_settings_are_cached():
    assert get_settings() is get_settings()


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_invalid_log_level_rejected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


def test_chunk_overlap_must_be_smaller_than_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


def test_embedding_endpoints_built_correctly():
    settings = get_settings()
    assert settings.embedding_endpoint.endswith(":embedContent")
    assert settings.embedding_batch_endpoint.endswith(":batchEmbedContents")
    assert settings.embedding_model in settings.embedding_endpoint
