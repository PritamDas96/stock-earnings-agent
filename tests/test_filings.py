"""Tests for SEC filing retrieval (httpx mocked)."""

from __future__ import annotations

import pytest

from mcp_server.tools import filings


class _HttpxResponse:
    def __init__(self, payload=None, text="", status=200):
        self._payload = payload or {}
        self.text = text
        self._status = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self._status >= 400:
            raise filings.httpx.HTTPStatusError("error", request=None, response=None)


@pytest.fixture(autouse=True)
def _clear_cik_cache():
    filings._ticker_to_cik_map.cache_clear()
    filings._sec_entries.cache_clear()
    yield
    filings._ticker_to_cik_map.cache_clear()
    filings._sec_entries.cache_clear()


def test_unsupported_form_type():
    assert "error" in filings.fetch_sec_filing("AAPL", "S-1")


def test_empty_ticker():
    assert "error" in filings.fetch_sec_filing("")


def test_list_companies_returns_sorted(monkeypatch):
    tickers = {
        "0": {"ticker": "MSFT", "cik_str": 789019, "title": "Microsoft Corp"},
        "1": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."},
    }

    def fake_get(url, headers, timeout):
        return _HttpxResponse(payload=tickers)

    monkeypatch.setattr(filings.httpx, "get", fake_get)
    companies = filings.list_companies()
    assert companies == [
        {"ticker": "AAPL", "name": "Apple Inc."},
        {"ticker": "MSFT", "name": "Microsoft Corp"},
    ]


def test_list_companies_empty_on_error(monkeypatch):
    def fake_get(url, headers, timeout):
        raise filings.httpx.HTTPError("network down")

    monkeypatch.setattr(filings.httpx, "get", fake_get)
    assert filings.list_companies() == []


def test_cik_not_found(monkeypatch):
    def fake_get(url, headers, timeout):
        return _HttpxResponse(payload={"0": {"ticker": "MSFT", "cik_str": 789019}})

    monkeypatch.setattr(filings.httpx, "get", fake_get)
    result = filings.fetch_sec_filing("NOPE", "10-Q")
    assert "CIK not found" in result["error"]


def test_successful_filing_fetch(monkeypatch):
    tickers = {"0": {"ticker": "AAPL", "cik_str": 320193}}
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q"],
                "accessionNumber": ["0000-00", "0001-11"],
                "primaryDocument": ["a.htm", "b.htm"],
                "filingDate": ["2026-01-01", "2026-04-01"],
            }
        }
    }

    def fake_get(url, headers, timeout):
        if "company_tickers" in url:
            return _HttpxResponse(payload=tickers)
        if "submissions" in url:
            return _HttpxResponse(payload=submissions)
        return _HttpxResponse(text="<html>FILING BODY</html>")

    monkeypatch.setattr(filings.httpx, "get", fake_get)
    result = filings.fetch_sec_filing("AAPL", "10-Q")
    assert result["form_type"] == "10-Q"
    assert result["filing_date"] == "2026-04-01"
    assert "FILING BODY" in result["text"]
    assert "b.htm" in result["filing_url"]
