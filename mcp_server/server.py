"""FastMCP server exposing stock-intelligence tools to MCP clients.

This module is a thin protocol adapter: each MCP tool delegates to a
framework-agnostic implementation in :mod:`mcp_server.tools`, so the same logic
is reused by the LangGraph agent. Business logic lives in the tool modules, not
here.

Run directly for stdio transport:

    python -m mcp_server.server
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core import get_logger
from mcp_server import tools

log = get_logger("mcp_server")

mcp = FastMCP("stock-intelligence")


@mcp.tool()
def get_financials(ticker: str) -> dict[str, Any]:
    """Fetch key fundamentals (revenue, margins, P/E, debt, growth) for a ticker.

    Args:
        ticker: Stock ticker symbol, e.g. AAPL, MSFT, TSLA.
    """
    return tools.get_financials(ticker)


@mcp.tool()
def get_price_history(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Get price trend, period high/low and performance for a ticker.

    Args:
        ticker: Stock ticker symbol.
        period: One of 1mo, 3mo, 6mo, 1y, 2y, 5y, max.
    """
    return tools.get_price_history(ticker, period)


@mcp.tool()
def calculate_ratios(ticker: str) -> dict[str, Any]:
    """Calculate valuation and health ratios (P/B, PEG, ROA, dividend yield).

    Args:
        ticker: Stock ticker symbol.
    """
    return tools.calculate_ratios(ticker)


@mcp.tool()
def get_analyst_recommendations(ticker: str) -> dict[str, Any]:
    """Get analyst consensus rating, price targets and recent rating changes.

    Args:
        ticker: Stock ticker symbol.
    """
    return tools.get_analyst_recommendations(ticker)


@mcp.tool()
def fetch_sec_filing(ticker: str, form_type: str = "10-Q") -> dict[str, Any]:
    """Fetch the latest SEC filing text (10-Q, 10-K or 8-K) for a company.

    Args:
        ticker: Stock ticker symbol, e.g. AAPL.
        form_type: 10-Q (quarterly), 10-K (annual) or 8-K (current report).
    """
    return tools.fetch_sec_filing(ticker, form_type)


@mcp.tool()
def search_earnings_documents(
    query: str, ticker: str | None = None, top_k: int = 5
) -> dict[str, Any]:
    """Semantic search over ingested earnings documents and filings (RAG).

    Args:
        query: Natural-language question about a company's earnings.
        ticker: Optional ticker to restrict results to one company.
        top_k: Number of passages to return.
    """
    return tools.search_earnings_documents(query, ticker, top_k)


def main() -> None:
    """Entry point for the stdio MCP server."""
    log.info("Starting stock-intelligence MCP server (stdio transport)")
    mcp.run()


if __name__ == "__main__":
    main()
