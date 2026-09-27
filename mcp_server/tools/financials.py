"""Market and fundamental data providers backed by yfinance.

These are plain, framework-agnostic functions returning JSON-serialisable
dicts. They are consumed both by the MCP server (as MCP tools) and by the
LangGraph agent (wrapped as LangChain tools), so business logic lives here
exactly once.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

from core import get_logger

log = get_logger("financials")

_VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}


def _clean_ticker(ticker: str) -> str:
    cleaned = (ticker or "").strip().upper()
    if not cleaned or not cleaned.replace(".", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return cleaned


def get_financials(ticker: str) -> dict[str, Any]:
    """Return key fundamental metrics for a ticker.

    Args:
        ticker: Stock ticker symbol, e.g. ``AAPL``.

    Returns:
        A dict of fundamentals, or ``{"error": ...}`` on failure.
    """
    try:
        symbol = _clean_ticker(ticker)
        info = yf.Ticker(symbol).info
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - external library, surface as data
        log.warning("get_financials({}) failed: {}", ticker, exc)
        return {"error": f"Failed to fetch financials for {ticker}: {exc}"}

    if not info or info.get("longName") is None and info.get("shortName") is None:
        return {"error": f"No financial data found for {ticker}"}

    return {
        "ticker": symbol,
        "company_name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "revenue": info.get("totalRevenue"),
        "gross_margins": info.get("grossMargins"),
        "operating_margins": info.get("operatingMargins"),
        "profit_margins": info.get("profitMargins"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "earnings_growth": info.get("earningsGrowth"),
        "revenue_growth": info.get("revenueGrowth"),
        "return_on_equity": info.get("returnOnEquity"),
        "free_cashflow": info.get("freeCashflow"),
    }


def get_price_history(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Return price performance metrics over a period.

    Args:
        ticker: Stock ticker symbol.
        period: One of ``1mo, 3mo, 6mo, 1y, 2y, 5y, max``.
    """
    try:
        symbol = _clean_ticker(ticker)
    except ValueError as exc:
        return {"error": str(exc)}

    if period not in _VALID_PERIODS:
        return {"error": f"Invalid period {period!r}. Choose from {sorted(_VALID_PERIODS)}"}

    try:
        hist = yf.Ticker(symbol).history(period=period)
    except Exception as exc:  # noqa: BLE001
        log.warning("get_price_history({}) failed: {}", ticker, exc)
        return {"error": f"Failed to fetch price history for {ticker}: {exc}"}

    if hist.empty:
        return {"error": f"No price data found for {ticker}"}

    current_price = round(float(hist["Close"].iloc[-1]), 2)
    start_price = round(float(hist["Close"].iloc[0]), 2)
    change_pct = (
        round((current_price - start_price) / start_price * 100, 2) if start_price else None
    )

    return {
        "ticker": symbol,
        "current_price": current_price,
        "start_price": start_price,
        "period_high": round(float(hist["Close"].max()), 2),
        "period_low": round(float(hist["Close"].min()), 2),
        "price_change_pct": change_pct,
        "period": period,
        "avg_daily_volume": int(hist["Volume"].mean()),
    }


def calculate_ratios(ticker: str) -> dict[str, Any]:
    """Return valuation and financial-health ratios for a ticker."""
    try:
        symbol = _clean_ticker(ticker)
        info = yf.Ticker(symbol).info
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        log.warning("calculate_ratios({}) failed: {}", ticker, exc)
        return {"error": f"Failed to fetch ratios for {ticker}: {exc}"}

    if not info:
        return {"error": f"No data found for {ticker}"}

    return {
        "ticker": symbol,
        "peg_ratio": info.get("pegRatio") or info.get("trailingPegRatio"),
        "price_to_book": info.get("priceToBook"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "return_on_assets": info.get("returnOnAssets"),
        "return_on_equity": info.get("returnOnEquity"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "enterprise_value": info.get("enterpriseValue"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
    }


def get_analyst_recommendations(ticker: str) -> dict[str, Any]:
    """Return analyst consensus rating, price targets and recent actions."""
    try:
        symbol = _clean_ticker(ticker)
        stock = yf.Ticker(symbol)
        info = stock.info
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        log.warning("get_analyst_recommendations({}) failed: {}", ticker, exc)
        return {"error": f"Failed to fetch recommendations for {ticker}: {exc}"}

    recommendations: dict[str, Any] = {
        "ticker": symbol,
        "analyst_rating": info.get("recommendationKey"),
        "target_price_mean": info.get("targetMeanPrice"),
        "target_price_high": info.get("targetHighPrice"),
        "target_price_low": info.get("targetLowPrice"),
        "number_of_analysts": info.get("numberOfAnalystOpinions"),
        "recent_ratings": [],
    }

    try:
        recs = stock.recommendations
        if recs is not None and not recs.empty:
            recommendations["recent_ratings"] = recs.tail(3).to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        log.debug("No recommendation history for {}: {}", symbol, exc)

    return recommendations
