"""Tests for the yfinance-backed financial providers (yfinance mocked)."""

from __future__ import annotations

import pandas as pd
import pytest

from mcp_server.tools import financials


class _FakeTicker:
    def __init__(self, info=None, history=None, recommendations=None):
        self._info = info or {}
        self._history = history if history is not None else pd.DataFrame()
        self._recommendations = recommendations

    @property
    def info(self):
        return self._info

    def history(self, period="1y"):
        return self._history

    @property
    def recommendations(self):
        return self._recommendations


@pytest.mark.parametrize("bad", ["", "  ", "@@@", "12$"])
def test_invalid_ticker_rejected(bad):
    assert "error" in financials.get_financials(bad)


def test_get_financials_maps_fields(monkeypatch):
    info = {"longName": "Apple Inc.", "sector": "Technology", "marketCap": 3_000_000_000_000,
            "trailingPE": 30.0, "profitMargins": 0.25}
    monkeypatch.setattr(financials.yf, "Ticker", lambda s: _FakeTicker(info=info))

    result = financials.get_financials("aapl")
    assert result["ticker"] == "AAPL"
    assert result["company_name"] == "Apple Inc."
    assert result["market_cap"] == 3_000_000_000_000
    assert result["pe_ratio"] == 30.0


def test_get_financials_empty_info_errors(monkeypatch):
    monkeypatch.setattr(financials.yf, "Ticker", lambda s: _FakeTicker(info={}))
    assert "error" in financials.get_financials("AAPL")


def test_get_price_history_computes_metrics(monkeypatch):
    hist = pd.DataFrame({"Close": [100.0, 110.0, 120.0], "Volume": [1000, 2000, 3000]})
    monkeypatch.setattr(financials.yf, "Ticker", lambda s: _FakeTicker(history=hist))

    result = financials.get_price_history("AAPL", "1y")
    assert result["current_price"] == 120.0
    assert result["start_price"] == 100.0
    assert result["period_high"] == 120.0
    assert result["period_low"] == 100.0
    assert result["price_change_pct"] == 20.0


def test_get_price_history_invalid_period():
    assert "error" in financials.get_price_history("AAPL", "13mo")


def test_get_price_history_empty(monkeypatch):
    monkeypatch.setattr(financials.yf, "Ticker", lambda s: _FakeTicker(history=pd.DataFrame()))
    assert "error" in financials.get_price_history("AAPL")


def test_calculate_ratios_peg_fallback(monkeypatch):
    info = {"trailingPegRatio": 1.5, "priceToBook": 8.0, "beta": 1.1}
    monkeypatch.setattr(financials.yf, "Ticker", lambda s: _FakeTicker(info=info))
    result = financials.calculate_ratios("AAPL")
    assert result["peg_ratio"] == 1.5
    assert result["price_to_book"] == 8.0


def test_analyst_recommendations_handles_missing_history(monkeypatch):
    info = {"recommendationKey": "buy", "targetMeanPrice": 200.0, "numberOfAnalystOpinions": 30}
    monkeypatch.setattr(
        financials.yf, "Ticker", lambda s: _FakeTicker(info=info, recommendations=None)
    )
    result = financials.get_analyst_recommendations("AAPL")
    assert result["analyst_rating"] == "buy"
    assert result["recent_ratings"] == []
