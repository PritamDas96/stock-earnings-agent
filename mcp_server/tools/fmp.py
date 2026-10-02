"""Financial Modeling Prep (FMP) fallback data provider.

``yfinance`` scrapes Yahoo Finance, which rate-limits/blocks shared datacenter
IPs (e.g. Streamlit Community Cloud) — so on hosted deployments
``yf.Ticker(...).info`` comes back empty and the UI shows "No financial data
found". FMP is a *keyed* REST API: it authenticates per request and is
therefore not IP-blocked, which makes it a reliable fallback in the cloud.

Uses FMP's current **stable** API (``/stable/<endpoint>?symbol=...``); the
legacy ``/api/v3`` endpoints were retired on 2025-08-31 and return 403 for keys
issued after that date.

Every public function returns a dict in the **same shape** as the matching
function in :mod:`mcp_server.tools.financials`, so it is a drop-in fallback, or
``None`` when FMP is unavailable (no key) or has no data for the symbol.

Enable by setting ``FMP_API_KEY`` (env var locally via ``.env``, or Streamlit
secrets on the cloud).
"""

from __future__ import annotations

import os
from typing import Any

import requests

from core import get_logger

try:  # core settings are optional here; env var is the primary source
    from core import get_settings
except Exception:  # noqa: BLE001  # pragma: no cover
    get_settings = None  # type: ignore[assignment]

log = get_logger("fmp")

_BASE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 15.0

# Approximate trading days per price period, for FMP's ``limit`` param.
_PERIOD_DAYS = {"1mo": 22, "3mo": 66, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260, "max": 5000}


def _key() -> str | None:
    """Return the FMP API key from the environment or settings, if configured."""
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key.strip() or None
    if get_settings is not None:
        try:
            value = getattr(get_settings(), "fmp_api_key", None)
            return value.strip() if value else None
        except Exception:  # noqa: BLE001 - settings may be missing other creds
            return None
    return None


def available() -> bool:
    """True when an FMP API key is configured."""
    return _key() is not None


def _get(endpoint: str, params: dict | None = None) -> Any:
    """GET a stable-API endpoint with the symbol/key attached; None on failure."""
    key = _key()
    if not key:
        return None
    query = dict(params or {})
    query["apikey"] = key
    try:
        resp = requests.get(f"{_BASE}/{endpoint}", params=query, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - external API, degrade gracefully
        log.warning("FMP {} failed: {}", endpoint, exc)
        return None


def _first(data: Any) -> dict[str, Any] | None:
    """First element of a list response (or the dict itself), else None."""
    if isinstance(data, list) and data:
        return data[0] if isinstance(data[0], dict) else None
    if isinstance(data, dict):
        return data
    return None


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def _sym(endpoint: str, symbol: str, extra: dict | None = None) -> Any:
    """Convenience: call a symbol-scoped stable endpoint."""
    params = {"symbol": symbol}
    if extra:
        params.update(extra)
    return _get(endpoint, params)


def get_financials(symbol: str) -> dict[str, Any] | None:
    """Fundamentals matching ``financials.get_financials`` output shape."""
    profile = _first(_sym("profile", symbol))
    if not profile:
        return None
    ratios = _first(_sym("ratios-ttm", symbol)) or {}
    keym = _first(_sym("key-metrics-ttm", symbol)) or {}
    growth = _first(_sym("financial-growth", symbol, {"limit": 1})) or {}
    income = _first(_sym("income-statement", symbol, {"period": "annual", "limit": 1})) or {}

    d_to_e = _num(ratios.get("debtToEquityRatioTTM"))
    return {
        "ticker": symbol,
        "company_name": profile.get("companyName"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "market_cap": _num(profile.get("marketCap")),
        "pe_ratio": _num(ratios.get("priceToEarningsRatioTTM")),
        "forward_pe": None,  # not available on FMP's free tier
        "revenue": _num(income.get("revenue")),
        "gross_margins": _num(ratios.get("grossProfitMarginTTM")),
        "operating_margins": _num(ratios.get("operatingProfitMarginTTM")),
        "profit_margins": _num(ratios.get("netProfitMarginTTM")),
        # yfinance reports debt/equity as a percentage (e.g. 78.4); FMP gives a
        # ratio (0.784). Scale so charts calibrated for yfinance stay consistent.
        "debt_to_equity": round(d_to_e * 100, 2) if d_to_e is not None else None,
        "current_ratio": _num(ratios.get("currentRatioTTM")),
        "earnings_growth": _num(growth.get("epsgrowth")),
        "revenue_growth": _num(growth.get("revenueGrowth")),
        "return_on_equity": _num(keym.get("returnOnEquityTTM")),
        "free_cashflow": _num(keym.get("freeCashFlowToEquityTTM")),
    }


def calculate_ratios(symbol: str) -> dict[str, Any] | None:
    """Valuation/health ratios matching ``financials.calculate_ratios``."""
    ratios = _first(_sym("ratios-ttm", symbol))
    keym = _first(_sym("key-metrics-ttm", symbol)) or {}
    profile = _first(_sym("profile", symbol)) or {}
    if not ratios and not keym:
        return None
    ratios = ratios or {}
    return {
        "ticker": symbol,
        "peg_ratio": _num(ratios.get("priceToEarningsGrowthRatioTTM")),
        "price_to_book": _num(ratios.get("priceToBookRatioTTM")),
        "price_to_sales": _num(ratios.get("priceToSalesRatioTTM")),
        "return_on_assets": _num(keym.get("returnOnAssetsTTM")),
        "return_on_equity": _num(keym.get("returnOnEquityTTM")),
        "dividend_yield": _num(ratios.get("dividendYieldTTM")),
        "beta": _num(profile.get("beta")),
        "enterprise_value": _num(keym.get("enterpriseValueTTM")),
        "ev_to_ebitda": _num(keym.get("evToEBITDATTM")),
    }


def _history(symbol: str, period: str) -> list[dict[str, Any]] | None:
    """Return FMP daily OHLCV rows oldest-first, trimmed to the period.

    The stable ``historical-price-eod/full`` endpoint ignores ``limit`` and
    returns the full available history (newest-first), so we slice to the most
    recent N trading days here.
    """
    days = _PERIOD_DAYS.get(period, 252)
    data = _sym("historical-price-eod/full", symbol)
    if not isinstance(data, list) or not data:
        return None
    recent = data[:days]  # newest N rows
    return list(reversed(recent))  # chart oldest->newest


def get_price_series(symbol: str, period: str) -> dict[str, Any] | None:
    """Raw OHLCV series matching ``financials.get_price_series``."""
    hist = _history(symbol, period)
    if not hist:
        return None
    return {
        "ticker": symbol,
        "period": period,
        "dates": [h.get("date") for h in hist],
        "open": [round(_num(h.get("open")) or 0.0, 2) for h in hist],
        "high": [round(_num(h.get("high")) or 0.0, 2) for h in hist],
        "low": [round(_num(h.get("low")) or 0.0, 2) for h in hist],
        "close": [round(_num(h.get("close")) or 0.0, 2) for h in hist],
        "volume": [int(_num(h.get("volume")) or 0) for h in hist],
    }


def get_price_history(symbol: str, period: str) -> dict[str, Any] | None:
    """Scalar price aggregates matching ``financials.get_price_history``."""
    hist = _history(symbol, period)
    if not hist:
        return None
    closes = [c for c in (_num(h.get("close")) for h in hist) if c is not None]
    volumes = [v for v in (_num(h.get("volume")) for h in hist) if v is not None]
    if not closes:
        return None
    current, start = round(closes[-1], 2), round(closes[0], 2)
    change = round((current - start) / start * 100, 2) if start else None
    return {
        "ticker": symbol,
        "current_price": current,
        "start_price": start,
        "period_high": round(max(closes), 2),
        "period_low": round(min(closes), 2),
        "price_change_pct": change,
        "period": period,
        "avg_daily_volume": int(sum(volumes) / len(volumes)) if volumes else 0,
    }


def get_financials_history(symbol: str, max_periods: int = 8) -> dict[str, Any] | None:
    """Quarterly income-statement trend matching ``financials.get_financials_history``."""
    # FMP's free tier caps income statements at 5 periods (402 beyond that),
    # so clamp the limit. Prefer quarterly; fall back to annual if unavailable.
    limit = min(max_periods, 5)
    rows = _sym("income-statement", symbol, {"period": "quarter", "limit": limit})
    if not isinstance(rows, list) or not rows:
        rows = _sym("income-statement", symbol, {"period": "annual", "limit": limit})
    if not isinstance(rows, list) or not rows:
        return None
    rows = list(reversed(rows))  # newest-first -> oldest-first

    def _margin(numer: Any, denom: Any) -> float | None:
        n, d = _num(numer), _num(denom)
        if n is None or d in (None, 0):
            return None
        return round(n / d * 100, 2)

    periods, revenue, net_income = [], [], []
    gross_margin, operating_margin, net_margin = [], [], []
    for r in rows:
        rev = _num(r.get("revenue"))
        ni = _num(r.get("netIncome"))
        date = r.get("date") or ""
        periods.append(date[:7] if date else str(r.get("fiscalYear", "")))
        revenue.append(rev)
        net_income.append(ni)
        gross_margin.append(_margin(r.get("grossProfit"), rev))
        operating_margin.append(_margin(r.get("operatingIncome"), rev))
        net_margin.append(_margin(ni, rev))

    if not any(v is not None for v in revenue):
        return None
    return {
        "ticker": symbol,
        "periods": periods,
        "revenue": revenue,
        "net_income": net_income,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
    }


def get_analyst_recommendations(symbol: str) -> dict[str, Any] | None:
    """Analyst consensus/targets matching ``financials.get_analyst_recommendations``.

    Best-effort: some analyst endpoints may be restricted on the free tier, in
    which case fields come back ``None`` and the UI degrades to "Analyst price
    targets unavailable".
    """
    consensus = _first(_sym("price-target-consensus", symbol))
    rating = _first(_sym("ratings-snapshot", symbol)) or {}
    if consensus is None and not rating:
        return None
    consensus = consensus or {}
    return {
        "ticker": symbol,
        "analyst_rating": rating.get("rating"),
        "target_price_mean": (
            _num(consensus.get("targetConsensus")) or _num(consensus.get("targetMedian"))
        ),
        "target_price_high": _num(consensus.get("targetHigh")),
        "target_price_low": _num(consensus.get("targetLow")),
        "number_of_analysts": None,
        "recent_ratings": [],
    }
