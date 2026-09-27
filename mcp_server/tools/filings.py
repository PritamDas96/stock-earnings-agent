"""SEC EDGAR filing retrieval.

Resolves a ticker to its CIK, finds the most recent filing of the requested
form type, and returns the raw document text (truncated). All requests carry a
descriptive User-Agent as required by SEC EDGAR access policy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx

from core import get_logger, get_settings

log = get_logger("filings")

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_MAX_TEXT_CHARS = 40_000
_VALID_FORMS = {"10-Q", "10-K", "8-K"}


@lru_cache(maxsize=1)
def _sec_entries() -> list[dict[str, Any]]:
    """Fetch and cache the raw SEC company directory for the process lifetime.

    Each entry has ``ticker``, ``cik_str`` and ``title`` (the company name).
    Cached with ``lru_cache``; exceptions are not cached, so a failed fetch is
    retried on the next call.
    """
    settings = get_settings()
    headers = {"User-Agent": settings.sec_user_agent}
    response = httpx.get(_TICKERS_URL, headers=headers, timeout=settings.http_timeout)
    response.raise_for_status()
    return list(response.json().values())


@lru_cache(maxsize=1)
def _ticker_to_cik_map() -> dict[str, str]:
    """Cache the SEC ticker to CIK mapping, derived from the company directory."""
    return {e["ticker"].upper(): str(e["cik_str"]).zfill(10) for e in _sec_entries()}


def list_companies() -> list[dict[str, str]]:
    """Return the SEC company directory as ``[{"ticker", "name"}, ...]``.

    Sorted by company name. Used to populate the company picker in the UI so
    users can search by name or symbol. Returns an empty list if the directory
    cannot be fetched, so callers can fall back gracefully.
    """
    try:
        entries = _sec_entries()
    except httpx.HTTPError as exc:
        log.warning("Failed to load SEC company directory: {}", exc)
        return []
    companies = [{"ticker": e["ticker"].upper(), "name": e["title"]} for e in entries]
    companies.sort(key=lambda c: c["name"].lower())
    return companies


def fetch_sec_filing(ticker: str, form_type: str = "10-Q") -> dict[str, Any]:
    """Fetch the most recent SEC filing of ``form_type`` for ``ticker``.

    Args:
        ticker: Stock ticker symbol, e.g. ``AAPL``.
        form_type: ``10-Q`` (quarterly), ``10-K`` (annual) or ``8-K``.

    Returns:
        A dict with ``filing_date``, ``filing_url`` and truncated ``text``, or
        ``{"error": ...}`` on failure.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return {"error": "Ticker symbol is required"}

    form = form_type.strip().upper()
    if form not in _VALID_FORMS:
        return {"error": f"Unsupported form_type {form_type!r}. Choose from {sorted(_VALID_FORMS)}"}

    settings = get_settings()
    headers = {"User-Agent": settings.sec_user_agent}

    try:
        cik = _ticker_to_cik_map().get(symbol)
    except httpx.HTTPError as exc:
        return {"error": f"Failed to fetch SEC ticker list: {exc}"}

    if not cik:
        return {"error": f"CIK not found for ticker {symbol}"}

    try:
        submissions = httpx.get(
            _SUBMISSIONS_URL.format(cik=cik), headers=headers, timeout=settings.http_timeout
        )
        submissions.raise_for_status()
        recent = submissions.json().get("filings", {}).get("recent", {})
    except httpx.HTTPError as exc:
        return {"error": f"Failed to fetch filings for {symbol}: {exc}"}

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])

    for i, filed_form in enumerate(forms):
        if filed_form != form:
            continue
        accession = accessions[i].replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_docs[i]}"
        )
        try:
            doc = httpx.get(filing_url, headers=headers, timeout=settings.http_timeout * 2)
            doc.raise_for_status()
        except httpx.HTTPError as exc:
            return {"error": f"Failed to fetch filing document: {exc}"}

        log.info("Fetched {} {} filed {}", symbol, form, dates[i])
        return {
            "ticker": symbol,
            "form_type": form,
            "filing_date": dates[i],
            "filing_url": filing_url,
            "text": doc.text[:_MAX_TEXT_CHARS],
            "truncated": len(doc.text) > _MAX_TEXT_CHARS,
        }

    return {"error": f"No {form} filing found for {symbol}"}
