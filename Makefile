# Developer task runner. On Windows without `make`, run the underlying commands
# directly (see README) or use `make` via Git Bash / Chocolatey.

PYTHON ?= python

.PHONY: help install install-dev test test-integration test-all lint format \
        healthcheck seed run mcp ingest clean

help:
	@echo "Targets:"
	@echo "  install           Install runtime dependencies"
	@echo "  install-dev       Install runtime + dev dependencies"
	@echo "  test              Run unit tests (Chroma integration tests excluded)"
	@echo "  test-integration  Run Chroma integration tests (own process)"
	@echo "  test-all          Run unit then integration tests"
	@echo "  lint              Ruff lint"
	@echo "  format            Ruff format + import sort"
	@echo "  healthcheck       Verify LLM, embeddings and vector store connectivity"
	@echo "  seed              Seed the demo corpus for evaluation"
	@echo "  run               Launch the Streamlit app"
	@echo "  mcp               Launch the MCP server (stdio)"
	@echo "  clean             Remove caches and the local vector store"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

test-integration:
	$(PYTHON) -m pytest tests/test_vector_store.py -m integration

test-all: test test-integration

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

healthcheck:
	$(PYTHON) -m scripts.healthcheck

seed:
	$(PYTHON) -m scripts.seed_eval_corpus

run:
	$(PYTHON) -m streamlit run app/main.py

mcp:
	$(PYTHON) -m mcp_server.server

clean:
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in ['.pytest_cache','.ruff_cache','chroma_db','htmlcov']]; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)]"
