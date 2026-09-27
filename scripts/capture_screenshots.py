"""Capture Streamlit UI and output screenshots for the project document.

Drives the running app with Playwright (Chromium) and writes PNGs to
docs/assets/. The app must already be running (default http://localhost:8600).

Usage:
    python -m scripts.capture_screenshots [base_url]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8600"
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def _settle(page, seconds: float = 2.0) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    time.sleep(seconds)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1024}, device_scale_factor=2)
        page.goto(BASE_URL, wait_until="load")
        page.wait_for_selector("text=Stock Earnings Agent", timeout=30000)
        _settle(page, 2)

        # 1) Snapshot tab, initial state.
        page.screenshot(path=str(OUT / "01_snapshot_input.png"), full_page=True)
        print("saved 01_snapshot_input.png")

        # 2) Snapshot output for AAPL.
        try:
            page.get_by_role("button", name="Fetch snapshot").click()
            page.wait_for_selector("text=Apple Inc.", timeout=45000)
            _settle(page, 2)
            page.screenshot(path=str(OUT / "02_snapshot_output.png"), full_page=True)
            print("saved 02_snapshot_output.png")
        except Exception as exc:
            print(f"snapshot output capture failed: {exc}")

        # 3) Ask the analyst tab + a completed answer.
        try:
            page.get_by_role("tab", name="Ask the analyst").click()
            _settle(page, 2)
            page.screenshot(path=str(OUT / "03_analyst_tab.png"), full_page=True)
            print("saved 03_analyst_tab.png")

            box = page.get_by_placeholder("e.g. How did AAPL")
            box.click()
            box.fill("In one sentence, what sector is AAPL in and its market cap?")
            box.press("Enter")
            # Wait for the assistant's answer to render.
            page.wait_for_selector("text=Bottom line", timeout=90000)
            _settle(page, 2)
            page.screenshot(path=str(OUT / "04_analyst_output.png"), full_page=True)
            print("saved 04_analyst_output.png")
        except Exception as exc:
            print(f"analyst capture partial: {exc}")
            try:
                page.screenshot(path=str(OUT / "04_analyst_output.png"), full_page=True)
                print("saved 04_analyst_output.png (state at timeout)")
            except Exception:
                pass

        # 5) Ingest tab.
        try:
            page.get_by_role("tab", name="Ingest documents").click()
            _settle(page, 2)
            page.screenshot(path=str(OUT / "05_ingest_tab.png"), full_page=True)
            print("saved 05_ingest_tab.png")
        except Exception as exc:
            print(f"ingest capture failed: {exc}")

        browser.close()


if __name__ == "__main__":
    main()
