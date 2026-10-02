# 📈 Stock Earnings Agent

A production-grade **equity earnings analysis agent** that combines live market
data, retrieval-augmented generation over earnings documents, and LLM reasoning —
exposed both as an interactive Streamlit app and as an **MCP server** that any
MCP client (e.g. Claude) can call.

```
 yfinance + SEC EDGAR + earnings PDFs
              │
   ┌──────────┴───────────┐
   ▼                      ▼
 rag_pipeline        mcp_server/tools   ← shared, framework-agnostic tools
 (chunk→embed→store)  (financials/filings/rag)
   │                      │
   ▼                      ▼
 ChromaDB            ┌──────────────┐
   │                 │ mcp_server   │ → MCP protocol (Claude, etc.)
   ▼                 └──────────────┘
 agents (LangGraph ReAct, Groq LLM) ──► app (Streamlit)
   │
   ▼
 evaluation (retrieval metrics + RAGAS)
```

## Features

- **6 financial tools** — fundamentals, price history, valuation ratios, analyst
  recommendations, SEC filing retrieval, and semantic search over ingested docs.
- **RAG pipeline** — PDF/text ingestion → recursive chunking → Google Gemini
  embeddings (via REST) → persistent ChromaDB store with metadata filtering.
- **LangGraph agent** — a tool-calling ReAct agent backed by Groq, with
  per-conversation memory.
- **MCP server** — the same tools exposed over the MCP protocol for Claude.
- **Streamlit UI** — quantitative snapshot, conversational analyst, and document
  ingestion.
- **Evaluation** — deterministic retrieval metrics (hit-rate\@k, MRR) plus a
  RAGAS harness for generation quality.
- **Tested** — unit tests (network mocked) + Chroma integration tests.

## Project layout

| Path | Responsibility |
|------|----------------|
| `core/` | Typed configuration (`pydantic-settings`) and logging (`loguru`) |
| `rag_pipeline/` | `embeddings`, `ingestion`, `chunking`, `vector_store`, `pipeline` |
| `mcp_server/tools/` | Framework-agnostic tool implementations (shared) |
| `mcp_server/server.py` | Thin FastMCP protocol adapter |
| `agents/` | `llm`, `tools`, `state`, `graph` (LangGraph ReAct agent) |
| `app/main.py` | Streamlit UI |
| `evaluation/` | Retrieval + RAGAS evaluation harness |
| `scripts/` | `ingest`, `healthcheck`, `seed_eval_corpus` CLIs |
| `tests/` | Pytest suite (`integration` marker for Chroma tests) |

## Setup

```powershell
# 1. Create and activate a virtual environment (Python 3.11+; 3.13 supported)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + test/lint tooling

# 3. Configure credentials
copy .env.example .env                    # then edit .env
```

Required in `.env`:

```
GROQ_API_KEY=<your_key>
GOOGLE_API_KEY=<your_key>
```

Verify everything is wired up:

```powershell
python -m scripts.healthcheck
```

## Usage

### Streamlit app
```powershell
streamlit run app/main.py
```

### MCP server (for Claude / MCP clients)
```powershell
python -m mcp_server.server
```
Register it with an MCP client, e.g.:
```json
{
  "mcpServers": {
    "stock-intelligence": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/Users/PritamDas/Documents/stock-earnings-agent"
    }
  }
}
```

### The agent from Python
```python
from agents import EarningsAgent

agent = EarningsAgent()
print(agent.analyze("How did NVDA's margins trend and what did management say?"))
```

### Ingest earnings documents
```powershell
python -m scripts.ingest ./earnings_docs --ticker AAPL --doc-type 10-Q
```

### Evaluate
```powershell
python -m scripts.seed_eval_corpus        # seed a small demo corpus
python -m evaluation.ragas_eval           # retrieval metrics (no LLM tokens)
```

## Testing

```powershell
python -m pytest                                  # unit tests
python -m pytest tests/test_vector_store.py -m integration   # Chroma tests
```

> **Why two commands?** ChromaDB's native (Rust) core conflicts with
> pandas/pyarrow when both run in the same process on Windows, causing a native
> crash. Chroma tests are marked `integration` and run in their own process.
> `make test-all` runs both.

## Configuration reference

All settings are environment variables (see `.env.example`). Notable ones:

| Variable | Default | Notes |
|----------|---------|-------|
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Must support tool calling |
| `LLM_MAX_TOKENS` | `800` | Keep within your Groq tier's per-request limit |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | 3072-dimensional |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Local persistent store |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Retrieval granularity |
| `RETRIEVAL_TOP_K` | `5` | Passages returned per query |

## Notes & limitations

- **Embeddings use the Google REST API directly** — `sentence-transformers`/
  PyTorch are deliberately avoided (DLL-incompatible with Python 3.13 on Windows).
- **Groq free tier** enforces a low output-tokens-per-minute limit; complex
  multi-step agent queries may hit `429`s. Upgrade to a paid tier and raise
  `LLM_MAX_TOKENS` for heavier use.
- This tool provides **analysis, not financial advice**.
