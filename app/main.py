"""Streamlit UI for the stock-earnings-agent.

Provides three panels:
  1. A quantitative snapshot (fundamentals, ratios, analyst view) for a ticker.
  2. A conversational analyst backed by the LangGraph agent.
  3. Document ingestion to grow the RAG knowledge base.

Run with:  streamlit run app/main.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure the project root is importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import get_settings  # noqa: E402
from mcp_server import tools  # noqa: E402

st.set_page_config(page_title="Stock Earnings Agent", page_icon="📈", layout="wide")


def _bootstrap_secrets() -> None:
    """Mirror Streamlit Cloud secrets into environment variables.

    On Streamlit Community Cloud there is no ``.env`` file; credentials are
    provided through the app's Secrets manager. Our configuration layer
    (``core.config``) reads ``os.environ``, so we copy any top-level secrets
    into the environment before settings are loaded. This keeps local (.env)
    and cloud (Secrets) deployments working from the same code.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file locally is fine
        return
    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)


_bootstrap_secrets()


@st.cache_resource(show_spinner=False)
def _load_agent():
    """Build the agent once per session (expensive to construct)."""
    from agents import EarningsAgent

    return EarningsAgent()


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from rag_pipeline import RAGPipeline

    return RAGPipeline()


# Well-known companies surfaced at the top of the picker for quick access.
POPULAR_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM", "V",
    "WMT", "JNJ", "UNH", "XOM", "MA", "PG", "HD", "COST", "KO", "PEP",
    "ADBE", "NFLX", "CRM", "AMD", "INTC", "CSCO", "DIS", "BAC", "ORCL", "PFE",
]


@st.cache_data(show_spinner="Loading company directory…", ttl=86_400)
def _load_directory() -> tuple[dict[str, str], list[str]]:
    """Return (name_by_ticker, ordered_tickers) from the SEC company directory.

    The ordered list pins popular tickers first, then the rest sorted by name.
    Cached for a day so the SEC file is fetched at most once per day per process.
    Returns empty structures if the directory cannot be loaded.
    """
    companies = tools.list_companies()  # [{"ticker", "name"}], sorted by name
    if not companies:
        return {}, []
    name_by_ticker = {c["ticker"]: c["name"] for c in companies}
    pinned = [t for t in POPULAR_TICKERS if t in name_by_ticker]
    pinned_set = set(pinned)
    rest = [c["ticker"] for c in companies if c["ticker"] not in pinned_set]
    return name_by_ticker, pinned + rest


def _company_picker(label: str, key: str, default: str = "AAPL", allow_none: bool = False) -> str:
    """Searchable company selector returning a ticker symbol (or "" if none).

    Falls back to a plain text input if the SEC directory is unavailable, so the
    app keeps working offline.
    """
    name_by_ticker, ordered = _load_directory()

    if not ordered:  # directory unavailable: degrade to free text
        value = st.text_input(f"{label} (symbol)", value="" if allow_none else default, key=key)
        return value.strip().upper()

    options = ([""] + ordered) if allow_none else ordered

    def _fmt(ticker: str) -> str:
        if ticker == "":
            return "— none —"
        name = name_by_ticker.get(ticker, "")
        return f"{name} ({ticker})" if name else ticker

    index = options.index(default) if (not allow_none and default in options) else 0
    return st.selectbox(label, options, index=index, format_func=_fmt, key=key)


def _check_config() -> bool:
    try:
        get_settings()
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Configuration error. GROQ_API_KEY and GOOGLE_API_KEY are not set.\n\n"
            "- Local: add them to a `.env` file in the project root.\n"
            "- Streamlit Cloud: open **Manage app**, then **Settings**, then "
            "**Secrets**, and add them in TOML format, for example:\n\n"
            '```toml\nGROQ_API_KEY = "your_key"\nGOOGLE_API_KEY = "your_key"\n```\n\n'
            f"Details:\n```\n{exc}\n```"
        )
        return False


def _snapshot_tab() -> None:
    st.subheader("Quantitative snapshot")
    st.caption("Search by company name or ticker symbol.")
    ticker = _company_picker("Company", key="snap_ticker", default="AAPL")
    period = st.selectbox("Price period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

    if not st.button("Fetch snapshot", type="primary"):
        return
    if not ticker:
        st.warning("Select a company.")
        return

    with st.spinner(f"Fetching data for {ticker}…"):
        financials = tools.get_financials(ticker)
        prices = tools.get_price_history(ticker, period)
        ratios = tools.calculate_ratios(ticker)
        analyst = tools.get_analyst_recommendations(ticker)

    if "error" in financials:
        st.error(financials["error"])
        return

    st.markdown(f"### {financials.get('company_name', ticker)}  \n"
                f"*{financials.get('sector') or '—'} · {financials.get('industry') or '—'}*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market cap", _human(financials.get("market_cap")))
    c2.metric("P/E (TTM)", _fmt(financials.get("pe_ratio")))
    c3.metric("Profit margin", _pct(financials.get("profit_margins")))
    c4.metric("Revenue growth", _pct(financials.get("revenue_growth")))

    if "error" not in prices:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${prices['current_price']}", f"{prices['price_change_pct']}%")
        c2.metric(f"{period} high", f"${prices['period_high']}")
        c3.metric(f"{period} low", f"${prices['period_low']}")
        c4.metric("Avg volume", _human(prices["avg_daily_volume"]))

    left, right = st.columns(2)
    with left:
        st.markdown("**Valuation ratios**")
        if "error" not in ratios:
            st.dataframe(
                _kv_rows(ratios, skip={"ticker"}),
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.markdown("**Analyst view**")
        if "error" not in analyst:
            st.dataframe(
                _kv_rows(analyst, skip={"ticker", "recent_ratings"}),
                use_container_width=True,
                hide_index=True,
            )


def _agent_tab() -> None:
    st.subheader("Ask the analyst")
    st.caption("The agent uses live financial data, SEC filings and your ingested documents.")

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    prompt = st.chat_input("e.g. How did AAPL's margins trend and what did management highlight?")
    if not prompt:
        return

    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                answer = _load_agent().analyze(prompt, thread_id="streamlit")
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ The agent hit an error: `{exc}`"
        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))


def _ingest_tab() -> None:
    st.subheader("Grow the knowledge base")
    st.caption("Upload earnings PDFs or transcripts to make them searchable by the agent.")

    ticker = _company_picker(
        "Associate with company (optional)", key="ingest_ticker", allow_none=True
    )
    doc_type = st.text_input("Document type", value="earnings_call")
    uploads = st.file_uploader(
        "PDF / text / markdown files",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )

    if st.button("Ingest", type="primary") and uploads:
        pipeline = _load_pipeline()
        metadata = {"doc_type": doc_type}
        if ticker:
            metadata["ticker"] = ticker
        total = 0
        with st.spinner("Ingesting…"), tempfile.TemporaryDirectory() as tmp:
            paths = []
            for upload in uploads:
                dest = Path(tmp) / upload.name
                dest.write_bytes(upload.getbuffer())
                paths.append(dest)
            total = pipeline.ingest_paths(paths, metadata=metadata)
        st.success(f"Ingested {total} chunk(s). Collection now holds {pipeline.store.count()}.")


# --- Formatting helpers --------------------------------------------------

def _fmt(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


def _pct(value) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def _human(value) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= size:
            return f"{value / size:.2f}{unit}"
    return str(value)


def _kv_rows(data: dict, skip: set[str]) -> list[dict]:
    return [{"metric": k, "value": v} for k, v in data.items() if k not in skip]


def main() -> None:
    st.title("📈 Stock Earnings Agent")
    if not _check_config():
        st.stop()

    snapshot, analyst, ingest = st.tabs(["Snapshot", "Ask the analyst", "Ingest documents"])
    with snapshot:
        _snapshot_tab()
    with analyst:
        _agent_tab()
    with ingest:
        _ingest_tab()


main()
