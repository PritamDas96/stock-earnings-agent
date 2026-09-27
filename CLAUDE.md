# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A production-grade stock earnings analysis agent that combines financial data (yfinance + SEC EDGAR), RAG over earnings documents (ChromaDB + Google embeddings), LLM reasoning (Groq / LangGraph), an MCP server exposing tools for Claude integration, and a Streamlit UI. All layers are implemented and tested.

## Environment Setup

```powershell
# Activate virtual environment (Python 3.11+, 3.13 supported)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + test/lint tooling
```

Required `.env` file at project root (copy from `.env.example`):
```
GROQ_API_KEY=<your_key>
GOOGLE_API_KEY=<your_key>
```

## Commands

```powershell
python -m scripts.healthcheck        # Verify LLM, embeddings, vector store connectivity
python -m pytest                     # Unit tests (Chroma integration tests excluded)
python -m pytest tests/test_vector_store.py -m integration   # Chroma integration tests
streamlit run app/main.py            # Streamlit UI
python -m mcp_server.server          # MCP server (stdio transport)
python -m scripts.ingest <path> --ticker AAPL   # Ingest documents into RAG store
python -m evaluation.ragas_eval      # Retrieval quality metrics
python -m ruff check .               # Lint
```

## Architecture

```
yfinance + SEC EDGAR + PDF earnings docs
        ↓
  core/                  ← pydantic-settings config + loguru logging (shared)
        ↓
  rag_pipeline/          ← ingest → chunk → embed (Google Gemini REST) → ChromaDB
        ↓
  mcp_server/tools/      ← framework-agnostic tool implementations (SHARED)
        ↓                        ├──► mcp_server/server.py (FastMCP → Claude)
  agents/                ← LangGraph ReAct agent (Groq) wraps the same tools
        ↓
  app/                   ← Streamlit UI
        ↓
  evaluation/            ← retrieval metrics + RAGAS harness
```

Key module responsibilities:
- **`core/`** — single source of truth for configuration (`get_settings()`) and logging (`get_logger()`). Logging goes to **stderr** (stdout is reserved for the MCP protocol).
- **`rag_pipeline/`** — `ingestion` (pdfplumber), `chunking` (recursive splitter), `embeddings` (direct REST to `gemini-embedding-001`), `vector_store` (ChromaDB), `pipeline` (orchestrator).
- **`mcp_server/tools/`** — pure functions returning JSON-serialisable dicts; reused by BOTH the MCP server and the agent. Errors are returned as `{"error": ...}`, never raised for expected failures.
- **`agents/`** — LangGraph `create_react_agent` backed by Groq; wraps the shared tools as LangChain tools.
- **`app/`** — Streamlit frontend.
- **`evaluation/`** — deterministic retrieval metrics + RAGAS.

## Key Decisions

- **LLM**: Groq API. Default model `qwen/qwen3.8-27b` (tool-calling capable and available on the current account; `llama-3.3-70b-versatile` is NOT available on this key). Configurable via `GROQ_MODEL`.
- **LLM tokens**: `LLM_MAX_TOKENS` defaults to 800 — the Groq free tier enforces a low output-tokens-per-minute limit. Raise on a paid tier.
- **Embeddings**: Direct REST API via `requests` using `gemini-embedding-001` (3072-d) — do NOT use `sentence-transformers` or PyTorch; both are DLL-incompatible with Python 3.13 on Windows.
- **Vector store**: ChromaDB (local, persistent). One `PersistentClient` per path per process (cached in `vector_store.py`) — constructing multiple clients crashes the native core on Windows.
- **Testing**: ChromaDB's native core conflicts with pandas/pyarrow in one process on Windows. Chroma tests are marked `integration` and run in a separate process; the default `pytest` run excludes them.
- **Orchestration**: LangGraph (not plain LangChain chains) for stateful agent workflows.
- **MCP**: FastMCP — the server is a thin adapter; business logic lives in `mcp_server/tools/`.
