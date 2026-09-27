"""LangChain tool wrappers around the shared tool implementations.

The same functions the MCP server exposes are wrapped as LangChain ``@tool``
objects so the LangGraph agent can call them. Descriptions are written for the
model — they explain *when* to reach for each tool.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from mcp_server import tools as impl


@tool
def get_financials(ticker: str) -> dict[str, Any]:
    """Get fundamental metrics (revenue, margins, P/E, debt, growth rates) for a
    company. Use for questions about profitability, size or financial health."""
    return impl.get_financials(ticker)


@tool
def get_price_history(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Get stock price performance over a period (1mo, 3mo, 6mo, 1y, 2y, 5y,
    max): current price, period high/low and percentage change."""
    return impl.get_price_history(ticker, period)


@tool
def calculate_ratios(ticker: str) -> dict[str, Any]:
    """Get valuation and health ratios (PEG, price-to-book, price-to-sales, ROA,
    ROE, dividend yield, beta, EV/EBITDA) for fundamental analysis."""
    return impl.calculate_ratios(ticker)


@tool
def get_analyst_recommendations(ticker: str) -> dict[str, Any]:
    """Get analyst consensus rating, mean/high/low price targets and recent
    rating changes to gauge market sentiment."""
    return impl.get_analyst_recommendations(ticker)


@tool
def fetch_sec_filing(ticker: str, form_type: str = "10-Q") -> dict[str, Any]:
    """Fetch the latest official SEC filing text (10-Q quarterly, 10-K annual,
    8-K current) for management discussion and risk factors."""
    return impl.fetch_sec_filing(ticker, form_type)


@tool
def search_earnings_documents(
    query: str, ticker: str | None = None, top_k: int = 5
) -> dict[str, Any]:
    """Semantic search over previously ingested earnings documents (call
    transcripts, reports). Use for qualitative questions about what management
    said. Returns cited passages."""
    return impl.search_earnings_documents(query, ticker, top_k)


def build_agent_tools() -> list:
    """Return the full toolset available to the earnings agent."""
    return [
        get_financials,
        get_price_history,
        calculate_ratios,
        get_analyst_recommendations,
        fetch_sec_filing,
        search_earnings_documents,
    ]
