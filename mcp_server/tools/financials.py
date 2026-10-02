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


def get_price_series(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Return the raw OHLCV time-series for charting (candlestick + volume).

    Unlike :func:`get_price_history`, which returns scalar aggregates for the
    agent/LLM, this returns the full daily arrays and is intended for the UI
    only (it is deliberately *not* registered as an agent/MCP tool to avoid
    flooding the model's context with hundreds of data points).

    Args:
        ticker: Stock ticker symbol.
        period: One of ``1mo, 3mo, 6mo, 1y, 2y, 5y, max``.

    Returns:
        ``{ticker, period, dates, open, high, low, close, volume}`` where each
        series is a list, or ``{"error": ...}`` on failure.
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
        log.warning("get_price_series({}) failed: {}", ticker, exc)
        return {"error": f"Failed to fetch price series for {ticker}: {exc}"}

    if hist.empty:
        return {"error": f"No price data found for {ticker}"}

    return {
        "ticker": symbol,
        "period": period,
        "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
        "open": [round(float(v), 2) for v in hist["Open"]],
        "high": [round(float(v), 2) for v in hist["High"]],
        "low": [round(float(v), 2) for v in hist["Low"]],
        "close": [round(float(v), 2) for v in hist["Close"]],
        "volume": [int(v) for v in hist["Volume"]],
    }


def get_financials_history(ticker: str, max_periods: int = 8) -> dict[str, Any]:
    """Return a quarterly income-statement trend for charting fundamentals.

    Pulls revenue, net income and derived margins from yfinance's quarterly
    income statement, oldest period first so it plots left-to-right. UI-only
    (not an agent/MCP tool) — the agent uses the scalar :func:`get_financials`.

    Args:
        ticker: Stock ticker symbol.
        max_periods: Most recent N quarters to return (chronological order).

    Returns:
        ``{ticker, periods, revenue, net_income, gross_margin,
        operating_margin, net_margin}`` or ``{"error": ...}`` on failure.
    """
    try:
        symbol = _clean_ticker(ticker)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        stmt = yf.Ticker(symbol).quarterly_income_stmt
    except Exception as exc:  # noqa: BLE001
        log.warning("get_financials_history({}) failed: {}", ticker, exc)
        return {"error": f"Failed to fetch financial history for {ticker}: {exc}"}

    if stmt is None or stmt.empty:
        return {"error": f"No quarterly financial history found for {ticker}"}

    def _row(*labels: str):
        """First matching income-statement row as a label->value lookup by column."""
        for label in labels:
            if label in stmt.index:
                return stmt.loc[label]
        return None

    revenue_row = _row("Total Revenue", "Operating Revenue")
    net_income_row = _row("Net Income", "Net Income Common Stockholders")
    gross_row = _row("Gross Profit")
    operating_row = _row("Operating Income", "Total Operating Income As Reported")

    if revenue_row is None:
        return {"error": f"No revenue line item found for {ticker}"}

    # Columns are period-end timestamps, newest first; chart oldest -> newest.
    columns = list(revenue_row.index)[::-1][-max_periods:]

    def _val(row, col):
        if row is None or col not in row.index:
            return None
        v = row[col]
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if f != f else f  # drop NaN

    def _margin(numer, denom):
        if numer is None or denom in (None, 0):
            return None
        return round(numer / denom * 100, 2)

    periods, revenue, net_income = [], [], []
    gross_margin, operating_margin, net_margin = [], [], []
    for col in columns:
        rev = _val(revenue_row, col)
        ni = _val(net_income_row, col)
        gross = _val(gross_row, col)
        op = _val(operating_row, col)
        periods.append(col.strftime("%Y-%m") if hasattr(col, "strftime") else str(col))
        revenue.append(rev)
        net_income.append(ni)
        gross_margin.append(_margin(gross, rev))
        operating_margin.append(_margin(op, rev))
        net_margin.append(_margin(ni, rev))

    return {
        "ticker": symbol,
        "periods": periods,
        "revenue": revenue,
        "net_income": net_income,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
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
