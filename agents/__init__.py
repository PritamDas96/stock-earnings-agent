"""LangGraph agent that reasons over financial data and earnings documents.

The agent is a tool-calling ReAct graph backed by Groq. It combines live market
data (yfinance), SEC filings, and RAG retrieval over ingested earnings
documents to answer analytical questions about a company's earnings.
"""

from agents.graph import EarningsAgent, build_agent
from agents.llm import get_llm
from agents.tools import build_agent_tools

__all__ = ["EarningsAgent", "build_agent", "get_llm", "build_agent_tools"]
