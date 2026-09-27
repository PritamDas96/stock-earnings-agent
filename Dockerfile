# syntax=docker/dockerfile:1
# Production image for the Streamlit UI. The MCP server can be run from the same
# image with:  docker run ... python -m mcp_server.server
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY core ./core
COPY rag_pipeline ./rag_pipeline
COPY agents ./agents
COPY mcp_server ./mcp_server
COPY app ./app
COPY evaluation ./evaluation
COPY scripts ./scripts

# Run as a non-root user.
RUN useradd --create-home appuser \
    && mkdir -p /app/chroma_db \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Readiness probe: verifies config, embeddings, vector store and LLM.
HEALTHCHECK --interval=60s --timeout=30s --start-period=20s --retries=3 \
    CMD python -m scripts.healthcheck || exit 1

CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
