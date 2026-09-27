"""Framework-agnostic tool implementations shared by the MCP server and agent.

Each function returns a JSON-serialisable dict and never raises for expected
failure modes (network errors, bad input) — errors are returned as
``{"error": ...}`` so both MCP clients and the agent can reason about them.
"""

from mcp_server.tools.filings import fetch_sec_filing, list_companies
from mcp_server.tools.financials import (
    calculate_ratios,
    get_analyst_recommendations,
    get_financials,
    get_price_history,
)
from mcp_server.tools.rag import search_earnings_documents

__all__ = [
    "get_financials",
    "get_price_history",
    "calculate_ratios",
    "get_analyst_recommendations",
    "fetch_sec_filing",
    "list_companies",
    "search_earnings_documents",
]
